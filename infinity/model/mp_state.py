"""Shared state for multiprocessing data parallelism.

Contains the SharedState container that holds all cross-process shared
memory tensors, communication queues, and synchronization primitives.

Uses 'spawn' start method to avoid CUDA fork guard issues. All state
is transferred via pickle — shared-memory tensors use shm handles for
zero-copy transfer; nn.Modules with shared-memory params are pickled
with their shm-backed storage.
"""

import ctypes
import logging
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


# Use spawn context throughout to avoid CUDA fork guard
_mp_ctx = mp.get_context('spawn')


class WorkerCommandType(Enum):
    FORWARD_BACKWARD = auto()
    SYNC_WEIGHTS = auto()
    SHUTDOWN = auto()


@dataclass
class WorkerCommand:
    type: WorkerCommandType
    # For FORWARD_BACKWARD:
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    global_valid_tokens: int = 0
    pixel_values: Optional[torch.Tensor] = None
    vision_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class WorkerResult:
    loss_val: float = 0.0
    total_tokens: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    valid_tokens: int = 0


class SharedState:
    """Cross-process shared state for multi-GPU data parallelism.

    All tensors are in shared memory (via share_memory_()). Modules have
    their parameters moved to shared memory so they can be pickled through
    spawn and arrive in workers backed by the same shm segments (zero-copy).
    """

    def __init__(self, cpu_master_model, config):
        """Build shared state from an initialized CPUMasterModel.

        Args:
            cpu_master_model: The CPUMasterModel with loaded weights
            config: CPUMasterConfig
        """
        self.config = config
        self.world_size = config.world_size
        self.device_ids = config.devices

        # --- Move all CPU master module params to shared memory ---
        # This makes them picklable through spawn with zero-copy shm handles
        for layer in cpu_master_model.cpu_layers:
            layer.share_memory()
        cpu_master_model.embedding.share_memory()
        if cpu_master_model.norm is not None:
            cpu_master_model.norm.share_memory()
        cpu_master_model.lm_head.share_memory()
        if cpu_master_model.rotary_emb is not None:
            cpu_master_model.rotary_emb.share_memory()

        # --- Shared-memory weight flat buffers (read by workers for H2D) ---
        self.layer_shared_flats = []
        for i, layer in enumerate(cpu_master_model.cpu_layers):
            numel = cpu_master_model.layer_numels[i]
            flat = torch.empty(numel, dtype=config.dtype)
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n
            flat.share_memory_()
            self.layer_shared_flats.append(flat)

        # --- Shared-memory gradient accumulators ---
        # Pre-allocate .grad for all unique params as shared memory.
        # Store grad tensors in a separate list because nn.Parameter.grad
        # is NOT preserved through pickle — workers call reattach_grads().
        self.grad_lock = _mp_ctx.Lock()
        self._param_grads = []  # parallel list to _all_param_refs()
        seen = set()
        for p in self._all_params_iter(cpu_master_model):
            if id(p) in seen:
                continue
            seen.add(id(p))
            grad = torch.zeros_like(p.data, device='cpu')
            grad.share_memory_()
            p.grad = grad
            self._param_grads.append(grad)

        # --- Model metadata (picklable scalars/lists) ---
        self.layer_numels = cpu_master_model.layer_numels
        self.layer_param_shapes = cpu_master_model.layer_param_shapes
        self.layer_param_numel = cpu_master_model.layer_param_numel
        self.layer_groups = cpu_master_model.layer_groups
        self.layer_to_group = cpu_master_model.layer_to_group
        self.max_layer_numel = cpu_master_model.max_layer_numel
        self.head_total_numel = cpu_master_model.head_total_numel
        self.embed_total_numel = cpu_master_model.embed_total_numel
        self._model_config = cpu_master_model._model_config
        self.vocab_size = cpu_master_model.vocab_size
        self.hidden_size = cpu_master_model.hidden_size
        self.num_heads = cpu_master_model.num_heads
        self.head_dim = cpu_master_model.head_dim
        self.tied_lm_head = cpu_master_model.tied_lm_head
        self.is_vlm = cpu_master_model.is_vlm
        self.layer_accepts_position_embeddings = cpu_master_model.layer_accepts_position_embeddings
        self.layer_accepts_position_ids = cpu_master_model.layer_accepts_position_ids
        self.layer_accepts_cache_position = cpu_master_model.layer_accepts_cache_position

        # --- CPU master modules (shared-memory params, picklable) ---
        self.cpu_layers = cpu_master_model.cpu_layers
        self.embedding = cpu_master_model.embedding
        self.norm = cpu_master_model.norm
        self.lm_head = cpu_master_model.lm_head
        self.rotary_emb = cpu_master_model.rotary_emb
        self.layer_cpu_params = cpu_master_model.layer_cpu_params

        # --- Communication queues (spawn-context) ---
        self.cmd_queues = [_mp_ctx.Queue() for _ in range(self.world_size)]
        self.result_queues = [_mp_ctx.Queue() for _ in range(self.world_size)]

    @staticmethod
    def _all_params_iter(model):
        """Iterate all trainable params in deterministic order (embedding, layers, norm, lm_head).

        Must match the order used in CPUMasterModel.get_parameters() for
        grad tensors to align after spawn pickling.
        """
        yield from model.embedding.parameters()
        for layer in model.cpu_layers:
            yield from layer.parameters()
        if model.norm is not None:
            yield from model.norm.parameters()
        yield from model.lm_head.parameters()

    def register_shared_flats_as_pinned(self):
        """Register shared-memory flat buffers with CUDA for pinned DMA speed.

        Shared-memory tensors aren't pinned by default, giving ~1.5x slower H2D.
        cudaHostRegister maps the pages into CUDA's address space, recovering
        full pinned DMA bandwidth (~21 GB/s vs ~14 GB/s unregistered).

        Must be called from a process that has initialized CUDA (i.e., workers,
        not the main process before spawning).
        """
        try:
            cuda_rt = ctypes.CDLL('libcudart.so')
        except OSError:
            logger.warning("Could not load libcudart.so — shared flats won't be pinned")
            return

        registered = 0
        for flat in self.layer_shared_flats:
            ptr = flat.data_ptr()
            nbytes = flat.nelement() * flat.element_size()
            # cudaHostRegisterPortable = 1 (accessible from all CUDA contexts)
            ret = cuda_rt.cudaHostRegister(
                ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1))
            if ret == 0:
                registered += 1

        if registered > 0:
            logger.info(f"Registered {registered}/{len(self.layer_shared_flats)} "
                        f"shared flats as CUDA pinned memory")
        else:
            logger.warning(f"Failed to register any shared flats as CUDA pinned memory")

    def reattach_grads(self):
        """Re-attach shared-memory grad tensors to params after spawn unpickling.

        nn.Parameter.grad is not preserved through pickle. Workers must call
        this after receiving SharedState to restore the grad references.
        """
        seen = set()
        grad_idx = 0
        for p in self._all_params_from_state():
            if id(p) in seen:
                continue
            seen.add(id(p))
            p.grad = self._param_grads[grad_idx]
            grad_idx += 1

    def _all_params_from_state(self):
        """Iterate all params from SharedState modules (same order as _all_params_iter)."""
        yield from self.embedding.parameters()
        for layer in self.cpu_layers:
            yield from layer.parameters()
        if self.norm is not None:
            yield from self.norm.parameters()
        yield from self.lm_head.parameters()

    def update_shared_flats(self):
        """Refresh shared-memory weight flats from CPU master params.

        Called by main process after optimizer.step().
        """
        for i, layer in enumerate(self.cpu_layers):
            flat = self.layer_shared_flats[i]
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n
