"""
MegaTrain Engine for VERL — CPU-memory-centric single-GPU training backend.

MegaTrain stores model parameters in CPU RAM and uses the GPU as a transient
compute engine with double-buffered weight transfer. This makes it possible
to train 100B+ parameter models on a single GPU for post-training workloads
(SFT, RLHF, DPO, GRPO) where memory is the bottleneck.

This engine wraps MegaTrain's CPUMasterModel to implement VERL's BaseEngine
interface, enabling it as a training backend for VERL's RL pipelines.
"""

import gc
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegaTrainEngine(BaseEngine):
    """
    Engine implementation using MegaTrain's CPU-memory-centric architecture.

    MegaTrain stores all parameters in CPU RAM and streams them to GPU
    layer-by-layer with double-buffered execution. This enables training
    models far larger than GPU memory (100B+ on a single GPU).

    Key differences from FSDP/Megatron backends:
    - Single-GPU, single-process (no torch.distributed required)
    - Parameters always live on CPU; GPU is transient compute
    - Forward and backward are fused (layer-by-layer with recompute)
    - No separate model.train()/eval() modes needed
    """

    def __init__(
        self,
        model_config,
        engine_config,
        optimizer_config,
        checkpoint_config,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.rank = 0  # Single-GPU, always rank 0

        self._is_offload_param = True  # MegaTrain always offloads params to CPU
        self._is_offload_optimizer = True  # Optimizer always on CPU

        self.cpu_master = None
        self.optimizer = None
        self.lr_scheduler = None

        # Mapping from HF state_dict key -> index in cpu_master.get_parameters()
        self._hf_key_to_param_idx = {}
        # Ordered list of HF keys matching cpu_master.get_parameters() order
        self._param_hf_keys = []

        # Frozen reference parameters for ref_in_actor mode.
        # Snapshot is taken after init; swap_to_ref/swap_to_actor exchange
        # the live CPU weights with this frozen copy for ref log_prob computation.
        self._ref_frozen_params = None  # list of CPU tensors, same order as get_parameters()

    @property
    def is_param_offload_enabled(self) -> bool:
        return True  # MegaTrain always keeps params on CPU

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return True  # Optimizer always on CPU

    def initialize(self):
        """Build the MegaTrain CPUMasterModel, optimizer, and LR scheduler."""
        from infinity.model.cpu_master import CPUMasterModel
        from infinity.config.training import CPUMasterConfig

        # Build MegaTrain config from VERL configs
        megatrain_config = self._build_megatrain_config()

        # Load HF model
        model_path = self.model_config.path if hasattr(self.model_config, 'path') else self.model_config.model_path
        trust_remote_code = getattr(self.model_config, 'trust_remote_code', True)
        dtype_str = getattr(self.engine_config, 'dtype', 'bfloat16')
        dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        is_vlm = hasattr(hf_config, 'vision_config')

        if is_vlm:
            try:
                from transformers import AutoModelForImageTextToText
                load_class = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForCausalLM
                load_class = AutoModelForCausalLM
        else:
            from transformers import AutoModelForCausalLM
            load_class = AutoModelForCausalLM

        attn_impl = getattr(self.engine_config, 'attn_implementation', 'flash_attention_2')

        logger.info(f"Loading model from {model_path} with attn={attn_impl}")
        hf_model = load_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )

        # Build HF param name mapping BEFORE creating CPUMasterModel
        # because CPUMasterModel decomposes the model into components
        self._build_param_name_mapping(hf_model)

        # Create CPUMasterModel
        self.cpu_master = CPUMasterModel(hf_model, megatrain_config)
        del hf_model
        gc.collect()

        # Create optimizer
        if not getattr(self.engine_config, 'forward_only', False):
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None

        # Immediately release GPU buffers after init.
        # VERL builds ref → actor → rollout (SGLang) in sequence.
        # If we keep GPU buffers alive, they consume ~6-8GB per model,
        # leaving insufficient GPU memory for SGLang to load the model.
        # Buffers will be rebuilt lazily before the first forward/backward.
        self.cpu_master.release_gpu_buffers()
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"MegaTrain engine initialized (GPU buffers released for colocated rollout). "
                     f"Model params on CPU, GPU device={megatrain_config.device}")

    def _build_param_name_mapping(self, hf_model):
        """Build mapping from HF param names to CPUMasterModel.get_parameters() order.

        Uses id(param) matching to guarantee exact correspondence between
        the ordered parameter list from get_parameters() and HF param names.

        CPUMasterModel.get_parameters(include_vision=False) returns params in:
          1. projector params (for VLM)
          2. embedding params
          3. layer 0..N-1 params
          4. norm params (if exists)
          5. lm_head params (if not tied)
        (rotary_emb.inv_freq is a buffer, not a parameter, so excluded)
        """
        # Build id(param) -> HF name mapping from the original HF model
        id_to_name = {}
        for name, param in hf_model.named_parameters():
            id_to_name[id(param)] = name

        # Simulate the exact iteration order of get_parameters(include_vision=False)
        # by walking the same components in the same order.
        from infinity.model.cpu_master import _discover_model_components
        components = _discover_model_components(hf_model)

        seen = set()
        param_keys = []

        def _add_module_params(module):
            if module is None:
                return
            for p in module.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    name = id_to_name.get(pid)
                    if name is not None:
                        param_keys.append(name)
                    else:
                        param_keys.append(f"param_{len(param_keys)}")

        # 1. projector (VLM) — get_parameters includes projector but not vision_encoder
        projector = components.get('projector')
        if components.get('is_vlm') and projector is not None:
            _add_module_params(projector)

        # 2. embedding
        _add_module_params(components['embedding'])

        # 3. layers (in order)
        for layer in components['layers']:
            _add_module_params(layer)

        # 4. norm
        _add_module_params(components.get('norm'))

        # 5. lm_head (skip if tied to embedding)
        lm_head = components['lm_head']
        emb = components['embedding']
        tied = False
        if hasattr(lm_head, "weight") and hasattr(emb, "weight"):
            tied = (lm_head.weight is emb.weight)
        if not tied:
            _add_module_params(lm_head)

        self._param_hf_keys = param_keys
        self._hf_key_to_param_idx = {k: i for i, k in enumerate(param_keys)}

        logger.info(f"Built HF param name mapping: {len(param_keys)} params "
                     f"(HF model has {len(id_to_name)} named_parameters)")

    def _build_megatrain_config(self):
        """Convert VERL configs to MegaTrain CPUMasterConfig."""
        from infinity.config.training import CPUMasterConfig

        model_path = self.model_config.path if hasattr(self.model_config, 'path') else self.model_config.model_path
        dtype_str = getattr(self.engine_config, 'dtype', 'bfloat16')
        dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16

        device = getattr(self.engine_config, 'device', 0)
        attn_impl = getattr(self.engine_config, 'attn_implementation', 'flash_attention_2')
        trust_remote_code = getattr(self.model_config, 'trust_remote_code', True)

        # Extract training hyperparams from optimizer config
        lr = getattr(self.optimizer_config, 'lr', 1e-5)
        weight_decay = getattr(self.optimizer_config, 'weight_decay', 0.01)
        max_grad_norm = getattr(self.optimizer_config, 'clip_grad', 1.0)
        max_seq_len = getattr(self.engine_config, 'max_seq_len', 2048)
        checkpoint_interval = getattr(self.engine_config, 'checkpoint_interval', 4)
        num_grad_slabs = getattr(self.engine_config, 'num_grad_slabs', 12)

        config = CPUMasterConfig(
            model_name=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
            dataset_path="__verl__",  # Placeholder; data comes from VERL
            dataset_name="",
            max_seq_len=max_seq_len,
            batch_size=1,  # VERL controls batching
            gradient_accumulation_steps=1,
            num_steps=1,
            learning_rate=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            checkpoint_interval=checkpoint_interval,
            num_grad_slabs=num_grad_slabs,
        )
        return config

    def _build_optimizer(self):
        """Build optimizer for CPU master parameters."""
        lr = getattr(self.optimizer_config, 'lr', 1e-5)
        weight_decay = getattr(self.optimizer_config, 'weight_decay', 0.01)
        betas = getattr(self.optimizer_config, 'betas', (0.9, 0.999))
        eps = getattr(self.optimizer_config, 'eps', 1e-8)

        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                self.cpu_master.get_parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adamw_mode=True,
            )
            logger.info("Using DeepSpeed CPUAdam optimizer")
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.cpu_master.get_parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            logger.info("Using PyTorch AdamW optimizer (DeepSpeed CPUAdam not available)")

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        """Build LR scheduler."""
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        total_steps = getattr(self.optimizer_config, 'total_training_steps', 1000)
        num_warmup_steps = getattr(self.optimizer_config, 'lr_warmup_steps', 0)
        lr_scheduler_type = getattr(self.optimizer_config, 'lr_scheduler_type', 'cosine')

        if num_warmup_steps <= 0:
            ratio = getattr(self.optimizer_config, 'lr_warmup_steps_ratio', 0.0)
            num_warmup_steps = int(ratio * total_steps)

        if lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = getattr(self.optimizer_config, 'min_lr_ratio', 0.01)
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)

    def train_mode(self, **kwargs):
        """Context manager for training mode.

        Returns BaseEngineCtx which calls engine.to("cuda") on enter (rebuilds
        GPU buffers) and engine.to("cpu") on exit (releases GPU buffers).
        This is critical for GPU memory time-sharing with the rollout engine.
        """
        return BaseEngineCtx(self, mode="train", **kwargs)

    def eval_mode(self, **kwargs):
        """Context manager for eval mode.

        Returns BaseEngineCtx which calls engine.to("cuda") on enter (rebuilds
        GPU buffers) and engine.to("cpu") on exit (releases GPU buffers).
        """
        return BaseEngineCtx(self, mode="eval", **kwargs)

    def snapshot_ref_params(self):
        """Snapshot current CPU parameters as frozen reference weights.

        Called once after model initialization. The snapshot is used by
        disable_adapter() to temporarily swap in reference weights for
        ref log_prob computation (ref_in_actor mode).
        """
        if self.cpu_master is None:
            raise RuntimeError("Cannot snapshot ref params before initialize()")
        self._ref_frozen_params = [
            p.data.clone() for p in self.cpu_master.get_parameters()
        ]
        logger.info(f"Snapshot {len(self._ref_frozen_params)} ref params "
                     f"({sum(p.numel() for p in self._ref_frozen_params) * 2 / 1e9:.1f} GB)")

    def disable_adapter(self):
        """Context manager that swaps in frozen reference weights.

        In ref_in_actor mode, VERL calls this when computing ref log_probs
        using the actor engine. For LoRA models, this disables adapters;
        for MegaTrain (full-param), this swaps CPU weights to the frozen
        reference copy and restores them on exit.

        Uses pointer swap (no memory copy) — O(num_params) pointer assignments,
        effectively free for any model size.
        """
        if self._ref_frozen_params is None:
            return nullcontext()

        engine = self

        @contextmanager
        def _swap_ctx():
            params = engine.cpu_master.get_parameters()
            # Pointer swap: exchange .data between actor and ref tensors
            for p, ref_p in zip(params, engine._ref_frozen_params):
                p.data, ref_p.data = ref_p.data, p.data
            # Update GPU copies (embedding, lm_head, norm) if buffers are live
            if not getattr(engine.cpu_master, '_gpu_released', True):
                engine.cpu_master._sync_params_to_gpu()
            try:
                yield
            finally:
                # Swap back: restore actor weights
                for p, ref_p in zip(params, engine._ref_frozen_params):
                    p.data, ref_p.data = ref_p.data, p.data
                if not getattr(engine.cpu_master, '_gpu_released', True):
                    engine.cpu_master._sync_params_to_gpu()

        return _swap_ctx()

    def _convert_nested_to_padded(self, input_ids, attention_mask):
        """Convert nested (jagged) tensors to padded tensors for MegaTrain.

        VERL's dataloader produces nested tensors (variable-length sequences packed
        via torch.nested). MegaTrain's CPUMasterModel expects regular [B, T] tensors.
        This method converts nested tensors to padded form and returns the offsets
        (cu_seqlens) needed to reconstruct nested tensors for the output.

        Returns:
            (input_ids, attention_mask, cu_seqlens)
            cu_seqlens is None if the inputs were already padded.
        """
        cu_seqlens = None
        if input_ids.is_nested:
            cu_seqlens = input_ids.offsets()
            batch_size = input_ids.size(0)
            max_seq_len = int(max(cu_seqlens.diff()))
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=0, output_size=(batch_size, max_seq_len)
            )
            if attention_mask.is_nested:
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )
            else:
                # attention_mask is a regular tensor; create a padded mask from cu_seqlens
                seq_lengths = cu_seqlens.diff()
                attention_mask = torch.zeros(
                    (batch_size, max_seq_len), dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                for i, sl in enumerate(seq_lengths):
                    attention_mask[i, :sl] = 1
        return input_ids, attention_mask, cu_seqlens

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """
        Run forward (and optionally backward) on a batch of data.

        For MegaTrain, forward and backward are fused: the CPU-offloaded
        layer-by-layer execution means we can't separate them.

        Args:
            data: TensorDict with input_ids, attention_mask, etc.
            loss_function: VERL loss function (PPO, DPO, etc.)
            forward_only: If True, only compute forward (inference mode).
        """
        # Ensure GPU buffers are available (may have been released for rollout)
        self.cpu_master.rebuild_gpu_buffers()

        # Set dp metadata required by VERL's loss functions (single GPU = dp_size 1)
        tu.assign_non_tensor(data, dp_size=1)
        if "loss_mask" in data.keys():
            loss_mask = data["loss_mask"]
            if loss_mask.is_nested:
                batch_num_tokens = loss_mask.values().sum().item()
            else:
                batch_num_tokens = loss_mask.sum().item()
            tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens)

        # Extract tensors from VERL's TensorDict
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        # Convert nested (jagged) tensors to padded tensors.
        # MegaTrain's CPUMasterModel expects regular [B, T] shaped tensors.
        input_ids, attention_mask, cu_seqlens = self._convert_nested_to_padded(
            input_ids, attention_mask
        )

        if forward_only:
            return self._forward_only(data, input_ids, attention_mask, loss_function, cu_seqlens)
        else:
            return self._forward_backward(data, input_ids, attention_mask, loss_function, cu_seqlens)

    def _forward_only(self, data, input_ids, attention_mask, loss_function, cu_seqlens=None):
        """Inference-only forward pass returning log_probs (and optionally entropy).

        Returns log_probs in the same format as FSDP engine:
        - If input was nested (cu_seqlens is not None): returns nested tensor
          with one log_prob per token, unshifted. no_padding_2_padding handles
          response extraction and left-shift.
        - If input was padded (cu_seqlens is None): returns padded [B, T-1].
        """
        calculate_entropy = tu.get_non_tensor_data(data=data, key="calculate_entropy", default=False)

        with torch.no_grad():
            logits = self.cpu_master.forward_logits(input_ids, attention_mask)
            logits_device = logits.device

            if cu_seqlens is not None:
                # Input was nested. Follow FSDP pattern:
                # 1. Narrow padded logits to valid lengths
                # 2. Flatten to (total_nnz, V)
                # 3. Compute log_probs with rolled flat input_ids
                # 4. Return as nested tensor with original cu_seqlens
                seq_lengths = cu_seqlens.diff().to(logits_device)
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                logits_nested = torch.nested.narrow(
                    logits, 1, starts, seq_lengths, layout=torch.jagged
                )
                logits_rmpad = torch.cat([t for t in logits_nested.unbind()])

                input_ids_orig = data["input_ids"]  # still nested from caller
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_orig.values(), shifts=-1, dims=0
                ).to(logits_device)

                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled
                )

                entropy = None
                if calculate_entropy:
                    entropy_rmpad = entropy_from_logits(logits_rmpad)
                    entropy = torch.nested.nested_tensor_from_jagged(
                        entropy_rmpad, cu_seqlens.to(logits_device)
                    )

                del logits, logits_nested, logits_rmpad

                log_probs = torch.nested.nested_tensor_from_jagged(
                    log_probs, cu_seqlens.to(logits_device)
                )
            else:
                # Input was already padded. Simple shifted log_probs.
                input_ids_gpu = input_ids.to(logits_device)
                log_probs = logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])

                entropy = None
                if calculate_entropy:
                    entropy = entropy_from_logits(logits)

                del logits

            gc.collect()
            torch.cuda.empty_cache()

            model_output = {"log_probs": log_probs}
            if entropy is not None:
                model_output["entropy"] = entropy

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=data, dp_group=None
                )
            else:
                loss = torch.tensor(0.0)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item() if isinstance(loss, torch.Tensor) else loss,
                "metrics": metrics,
            }

            return output

    def _forward_backward(self, data, input_ids, attention_mask, loss_function, cu_seqlens=None):
        """Training forward+backward with VERL loss function.

        Adapts VERL's loss_function(model_output, data, dp_group) interface
        to MegaTrain's loss_fn(logits, input_ids_gpu) interface.
        """
        def _loss_fn_adapter(logits, input_ids_gpu):
            """Adapt VERL's loss function to MegaTrain's interface.

            MegaTrain's forward_and_backward_custom_loss expects:
                loss_fn(logits: [B,T,V], input_ids_gpu: [B,T]) -> (loss, meta)

            VERL's loss function expects:
                loss_fn(model_output={"log_probs": ...}, data=TensorDict, dp_group=None) -> (loss, metrics)
            """
            logits_device = logits.device

            if cu_seqlens is not None:
                # Extract per-sample logits, compute log_probs on flat tokens,
                # then reconstruct as nested tensor for VERL's loss function.
                seq_lengths = cu_seqlens.diff().to(logits_device)
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                logits_nested = torch.nested.narrow(
                    logits, 1, starts, seq_lengths, layout=torch.jagged
                )
                logits_rmpad = torch.cat([t for t in logits_nested.unbind()])
                input_ids_orig = data["input_ids"]  # still nested
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_orig.values(), shifts=-1, dims=0
                ).to(logits_device)
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled
                )
                log_probs = torch.nested.nested_tensor_from_jagged(
                    log_probs, cu_seqlens.to(logits_device)
                )
            else:
                log_probs = logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])

            model_output = {"log_probs": log_probs}

            # Move only loss-related tensor fields to GPU to avoid OOM.
            # The loss function accesses: response_mask, old_log_probs, advantages,
            # and optionally rollout_is_weights, ref_log_prob.
            # Non-tensor metadata (dp_size, batch_num_tokens, etc.) stays as-is.
            loss_fields = ["response_mask", "old_log_probs", "advantages", "loss_mask"]
            for opt_field in ["rollout_is_weights", "ref_log_prob"]:
                if opt_field in data.keys():
                    loss_fields.append(opt_field)
            for field in loss_fields:
                if field in data.keys():
                    val = data[field]
                    if isinstance(val, torch.Tensor):
                        data[field] = val.to(logits_device)

            loss, metrics = loss_function(
                model_output=model_output, data=data, dp_group=None
            )
            return loss, metrics

        loss_val, num_tokens, timing, meta = self.cpu_master.forward_and_backward_custom_loss(
            input_ids, attention_mask, _loss_fn_adapter
        )

        output = {
            "model_output": {"log_probs": None},  # Already consumed by loss
            "loss": loss_val,
            "metrics": meta if meta else {},
        }

        return output

    # train_batch() is inherited from BaseEngine which already does:
    #   zero_grad -> forward_backward_batch -> optimizer_step
    # No need to override.

    def optimizer_zero_grad(self):
        """Zero gradients on CPU master parameters."""
        self.cpu_master.zero_grad()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        """Clip gradients, step optimizer, sync params to GPU."""
        clip_grad = getattr(self.optimizer_config, 'clip_grad', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.cpu_master.get_parameters(), clip_grad
        )
        if self.optimizer is not None:
            self.optimizer.step()
        self.cpu_master._sync_params_to_gpu()
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def lr_scheduler_step(self):
        """Step the LR scheduler."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return 0.0

    def get_per_tensor_param(self, **kwargs):
        """Yield (name, tensor) pairs for all model parameters.

        Returns a generator of (hf_key, param_tensor) tuples that uses
        HuggingFace-compatible parameter names. This is required for
        weight sync between the training engine and rollout engine.

        Tensors are moved to CUDA because SGLang's weight sync serialization
        expects CUDA tensors (its _reduce_tensor patch accesses device index).

        Returns:
            (generator of (name: str, tensor: torch.Tensor), peft_config or None)
        """
        params = self.cpu_master.get_parameters()
        device = torch.device(f"cuda:{get_device_id()}")

        def _param_generator():
            for idx, param in enumerate(params):
                if idx < len(self._param_hf_keys):
                    name = self._param_hf_keys[idx]
                else:
                    # Fallback for any unmapped params
                    name = f"param_{idx}"
                yield (name, param.data.to(device, non_blocking=True))

        return _param_generator(), None

    def get_data_parallel_size(self):
        return 1  # Single GPU

    def get_data_parallel_rank(self):
        return 0  # Single GPU

    def get_data_parallel_group(self):
        return None  # No distributed group

    def is_mp_src_rank_with_outputs(self):
        return True  # Single GPU, always has outputs

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Release or rebuild GPU buffers for MegaTrain.

        MegaTrain keeps all parameters on CPU. The GPU only holds transient
        compute buffers (double-buffered flat params, layer templates,
        embedding/norm/lm_head copies). When device='cpu', release these
        buffers so colocated inference engines (SGLang) can use the GPU.
        When device='cuda', rebuild them before training resumes.
        """
        if self.cpu_master is None:
            return

        if device == "cpu" or device == torch.device("cpu"):
            self.cpu_master.release_gpu_buffers()
            gc.collect()
            torch.cuda.empty_cache()
        elif "cuda" in str(device):
            self.cpu_master.rebuild_gpu_buffers()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ):
        """Save model checkpoint using HF-compatible parameter names."""
        os.makedirs(local_path, exist_ok=True)
        params = self.cpu_master.get_parameters()
        state = {}
        for idx, param in enumerate(params):
            if idx < len(self._param_hf_keys):
                name = self._param_hf_keys[idx]
            else:
                name = f"param_{idx}"
            state[name] = param.data.clone()

        save_dict = {
            'model_state_dict': state,
            'global_step': global_step,
        }
        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(save_dict, os.path.join(local_path, f"checkpoint_{global_step}.pt"))
        logger.info(f"Saved checkpoint at step {global_step} to {local_path}")

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: bool = True,
        **kwargs,
    ):
        """Load model checkpoint."""
        import glob as glob_mod
        ckpt_file = os.path.join(local_path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            ckpts = sorted(glob_mod.glob(os.path.join(local_path, "checkpoint_*.pt")))
            if ckpts:
                ckpt_file = ckpts[-1]
            else:
                logger.warning(f"No checkpoint found in {local_path}")
                return

        state = torch.load(ckpt_file, map_location="cpu")
        model_state = state.get('model_state_dict', state.get('model_state', {}))

        params = self.cpu_master.get_parameters()
        for idx, param in enumerate(params):
            if idx < len(self._param_hf_keys):
                name = self._param_hf_keys[idx]
            else:
                name = f"param_{idx}"
            if name in model_state:
                param.data.copy_(model_state[name])

        if self.optimizer is not None:
            opt_state = state.get('optimizer_state_dict', state.get('optimizer_state'))
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)

        self.cpu_master._sync_params_to_gpu()
        logger.info(f"Loaded checkpoint from {ckpt_file}")

    def cleanup(self):
        """Clean up MegaTrain resources."""
        if self.cpu_master is not None:
            self.cpu_master.cleanup()


@EngineRegistry.register(
    model_type="language_model",
    backend="megatrain",
    device="cuda",
)
class MegaTrainEngineWithLMHead(MegaTrainEngine):
    """MegaTrain engine registered for language model training with VERL."""
    pass
