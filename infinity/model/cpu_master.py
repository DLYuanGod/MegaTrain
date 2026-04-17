"""CPU Master Model with explicit recompute and async pipeline.

This module implements a CPU-backed training system for large language models that
exceed GPU memory capacity. Key features:
- FP32 master parameters stored on CPU
- Double-buffered GPU layer execution
- Async weight transfer and gradient collection
- K-slab gradient pool for memory efficiency
- Manual gradient computation without autograd overhead
- Multi-GPU data parallelism via multiprocessing

Supports any HuggingFace decoder-only model architecture:
- Standard dense models (Llama, Qwen, Mistral, Phi, Gemma, etc.)
- Hybrid attention models (Qwen3.5 linear+full attention)
- MoE models (Mixtral, DeepSeek-MoE, Qwen3-Next)
"""

import inspect
import logging
import copy
import gc
import threading
import queue
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from infinity.config.training import CPUMasterConfig

logger = logging.getLogger(__name__)

# Try to import flash-attn CrossEntropyLoss
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
    FLASH_CE_AVAILABLE = True
except ImportError:
    FLASH_CE_AVAILABLE = False


def _preserve_attn_implementation(layer, model_config):
    """Ensure Flash Attention implementation is preserved when layer is moved to GPU.

    HuggingFace may reset _attn_implementation during .to(device) or deepcopy.
    This explicitly sets the attention config on the layer's attention module.
    """
    attn_impl = getattr(model_config, '_attn_implementation', None)
    if attn_impl is None:
        return

    # Walk the layer to find attention modules and set their config
    for name, module in layer.named_modules():
        if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
            module.config._attn_implementation = attn_impl
        # Some models store it directly on the attention module
        if hasattr(module, '_attn_implementation'):
            module._attn_implementation = attn_impl


def _discover_model_components(hf_model):
    """Discover model components via attribute introspection.

    Supports LLM and VLM models:
    - LLM: LLaMA, Qwen, Mistral, Phi, Gemma, GPT-2, DeepSeek, etc.
    - VLM: Qwen2-VL, Qwen3-VL, Qwen3.5-VL, LLaVA, Llama4-VL, Gemma3-VL,
            InternVL, GLM-4V, MiniCPM-V, etc.

    Returns:
        dict with keys: 'model_core', 'embedding', 'layers', 'norm', 'lm_head',
                        'rotary_emb', 'vision_encoder', 'projector', 'is_vlm'
    """
    model_type = getattr(hf_model.config, 'model_type', '')

    # === VLM Detection & Component Extraction ===
    # VLM models wrap a language model + vision encoder + projector
    # We need to find the language model first, then extract LLM components from it
    VLM_CONFIGS = {
        # model_type: (language_model_attr, vision_attrs, projector_attr)
        # language_model_attr: dot-separated path from hf_model to the language model root
        'qwen2_vl':    ('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'qwen2_5_vl':  ('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'qwen3_vl':    ('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'qwen3_vl_moe':('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'qwen3_5':     ('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'qwen3_5_moe': ('model.language_model',  ['model.visual'],  'model.visual.merger'),
        'llava':       ('language_model',         ['vision_tower'],  'multi_modal_projector'),
        'llava_next':  ('language_model',         ['vision_tower'],  'multi_modal_projector'),
        'llama4':      ('language_model',         ['vision_model'],  'multi_modal_projector'),
        'gemma3':      ('language_model',         ['vision_tower'],  'multi_modal_projector'),
        'internvl':    ('language_model',         ['vision_tower'],  'multi_modal_projector'),
        'glm4v':       ('model.language_model',   ['model.visual'],  'model.visual.merger'),
        'glm4v_moe':   ('model.language_model',   ['model.visual'],  'model.visual.merger'),
        'minicpmv':    ('llm',                    ['vpm'],           'resampler'),
        'minicpmo':    ('llm',                    ['vpm', 'apm'],    'resampler'),
        'mllama':      ('language_model',         ['vision_model'],  'multi_modal_projector'),
        'paligemma':   ('language_model',         ['vision_tower'],  'multi_modal_projector'),
    }

    is_vlm = False
    vision_encoder = None
    projector = None
    lm_root = hf_model  # For LLMs, search from the top-level model

    # Check if this is a VLM by model_type or by presence of vision components
    if model_type in VLM_CONFIGS:
        lm_attr, vision_attrs, proj_attr = VLM_CONFIGS[model_type]

        # Extract vision encoder (may be dot-separated paths)
        vision_parts = []
        for va in vision_attrs:
            v = hf_model
            for key in va.split('.'):
                v = getattr(v, key, None)
                if v is None:
                    break
            if v is not None:
                vision_parts.append(v)
        if vision_parts:
            # Wrap multiple vision parts in a ModuleList for unified handling
            if len(vision_parts) == 1:
                vision_encoder = vision_parts[0]
            else:
                vision_encoder = nn.ModuleList(vision_parts)
            is_vlm = True
            logger.info(f"VLM detected ({model_type}): vision_encoder from {vision_attrs}")

        # Extract projector
        p = hf_model
        for key in proj_attr.split('.'):
            p = getattr(p, key, None)
            if p is None:
                break
        if p is not None:
            projector = p
            logger.info(f"VLM projector at: {proj_attr}")

        # For VLMs, the language model is nested (may be dot-separated path)
        lm_root = hf_model
        for key in lm_attr.split('.'):
            lm_root = getattr(lm_root, key, None)
            if lm_root is None:
                break
        if lm_root is None:
            lm_root = hf_model
            logger.warning(f"VLM language model attr '{lm_attr}' not found, using top-level model")
    elif hasattr(hf_model.config, 'vision_config'):
        # Generic VLM detection via config
        logger.info(f"Detected vision_config in model config, attempting VLM discovery")
        for lm_attr in ['language_model', 'model', 'llm']:
            candidate = getattr(hf_model, lm_attr, None)
            if candidate is not None and hasattr(candidate, 'layers') or hasattr(candidate, 'model'):
                lm_root = candidate
                break
        for va in ['vision_tower', 'visual', 'vision_model', 'vpm']:
            v = getattr(hf_model, va, None)
            if v is not None:
                vision_encoder = v
                is_vlm = True
                logger.info(f"VLM vision_encoder found at: {va}")
                break
        for pa in ['multi_modal_projector', 'visual.merger', 'resampler']:
            p = hf_model
            for key in pa.split('.'):
                p = getattr(p, key, None)
                if p is None:
                    break
            if p is not None:
                projector = p
                logger.info(f"VLM projector found at: {pa}")
                break

    # === LLM Component Discovery (from lm_root) ===
    # For VLMs, lm_root is the language model; for LLMs, it's the top-level model
    model_core = getattr(lm_root, 'model', lm_root)

    # Find embedding (search from both hf_model and lm_root for VLM compatibility)
    EMBED_PATHS = [
        ('model', 'embed_tokens'), ('transformer', 'wte'),
        ('model', 'decoder', 'embed_tokens'), ('embed_tokens',),
    ]
    embedding = None
    for search_root in ([lm_root, hf_model] if is_vlm else [hf_model]):
        for path in EMBED_PATHS:
            obj = search_root
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                embedding = obj
                logger.info(f"Found embedding at: {'.'.join(path)}")
                break
        if embedding is not None:
            break
    if embedding is None:
        raise AttributeError("Could not find embedding layer")

    # Find layers (search from lm_root for VLMs)
    LAYER_PATHS = [
        ('model', 'layers'), ('transformer', 'h'),
        ('model', 'decoder', 'layers'), ('decoder', 'layers'), ('layers',),
    ]
    layers = None
    for search_root in ([lm_root, hf_model] if is_vlm else [hf_model]):
        for path in LAYER_PATHS:
            obj = search_root
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, '__len__') and len(obj) > 0:
                layers = list(obj)
                logger.info(f"Found {len(layers)} layers at: {'.'.join(path)}")
                break
        if layers is not None:
            break
    if layers is None:
        raise AttributeError("Could not find decoder layers")

    # Find final norm (search from lm_root for VLMs)
    NORM_PATHS = [
        ('model', 'norm'), ('transformer', 'ln_f'),
        ('model', 'decoder', 'final_layer_norm'), ('norm',),
    ]
    norm = None
    for search_root in ([lm_root, hf_model] if is_vlm else [hf_model]):
        for path in NORM_PATHS:
            obj = search_root
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                norm = obj
                logger.info(f"Found final_norm at: {'.'.join(path)}")
                break
        if norm is not None:
            break

    # Find lm_head (search from lm_root and hf_model)
    lm_head = None
    for search_root in ([lm_root, hf_model] if is_vlm else [hf_model]):
        lm_head = getattr(search_root, 'lm_head', None)
        if lm_head is not None:
            break
    if lm_head is None:
        lm_head = getattr(getattr(hf_model, 'transformer', None), 'lm_head', None)
    if lm_head is None:
        raise AttributeError("Could not find lm_head")

    # Find rotary_emb (model-level, modern HF models)
    rotary_emb = getattr(model_core, 'rotary_emb', None)

    return {
        'model_core': model_core,
        'embedding': embedding,
        'layers': layers,
        'norm': norm,
        'lm_head': lm_head,
        'rotary_emb': rotary_emb,
        'vision_encoder': vision_encoder,
        'projector': projector,
        'is_vlm': is_vlm,
    }


def _introspect_layer_forward(layer):
    """Introspect a layer's forward signature to determine accepted kwargs.

    Returns a set of accepted parameter names.
    """
    try:
        sig = inspect.signature(layer.forward)
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        # Fallback: assume modern HF signature
        return {'hidden_states', 'attention_mask', 'position_ids',
                'position_embeddings', 'cache_position',
                'use_cache', 'output_attentions'}


def _group_layers_by_structure(cpu_layers):
    """Group layers by their parameter structure (name, shape tuples).

    Returns:
        layer_groups: dict {group_id: {'param_structure': [...], 'numel': int, 'indices': [...]}}
        layer_to_group: list mapping layer_idx -> group_id
    """
    layer_groups = {}
    layer_to_group = []
    structure_to_group_id = {}

    for i, layer in enumerate(cpu_layers):
        structure = tuple((name, tuple(p.shape)) for name, p in layer.named_parameters())
        structure_key = hash(structure)

        if structure_key not in structure_to_group_id:
            group_id = len(layer_groups)
            structure_to_group_id[structure_key] = group_id
            layer_groups[group_id] = {
                'param_structure': structure,
                'numel': sum(p.numel() for p in layer.parameters()),
                'indices': [],
                'param_shapes': [p.shape for p in layer.parameters()],
                'param_numels': [p.numel() for p in layer.parameters()],
            }
        else:
            group_id = structure_to_group_id[structure_key]

        layer_groups[group_id]['indices'].append(i)
        layer_to_group.append(group_id)

    return layer_groups, layer_to_group


class _GPUContext:
    """Holds all per-GPU state for multi-GPU data parallelism."""
    __slots__ = [
        'gpu_idx', 'device',
        'gpu_flat_buffers', 'cpu_flat_buffers',
        'gpu_layer_templates',
        'emb_gpu', 'norm_gpu', 'lm_head_gpu', 'rotary_gpu',
        'compute_stream', 'weight_stream', 'grad_stream',
        'weight_ready_events', 'h2d_done_events', 'backward_done_events',
        'buffer_busy_events', 'buffer_free_events', 'template_free_events',
        'param_sync_event', 'loss_backward_done', 'embedding_backward_done',
        'layer_grad_slabs', 'layer_slab_events', 'layer_slab_free_list',
        'head_grad_slab', 'head_slab_event', 'head_slab_free',
        'embed_grad_slab', 'embed_slab_event', 'embed_slab_free',
        'ce_loss',
    ]

    def __init__(self, gpu_idx, device):
        self.gpu_idx = gpu_idx
        self.device = device


class CPUMasterModel:
    """CPU master with explicit recompute - TRUE async pipeline.

    Supports any HuggingFace decoder-only model and VLM. Handles:
    - Uniform layers (Llama, Qwen2, Mistral, etc.)
    - Hybrid attention (Qwen3.5 linear+full)
    - MoE layers (Mixtral, DeepSeek, Qwen3-Next)
    - VLM (Qwen2-VL, Qwen3-VL, LLaVA, Gemma3-VL, InternVL, etc.)
    - Multi-GPU data parallelism
    """

    def __init__(self, hf_model, config: CPUMasterConfig):
        self.config = config
        self.world_size = config.world_size

        # === Discover model structure (model-agnostic, LLM + VLM) ===
        components = _discover_model_components(hf_model)

        # VLM components (CPU offloaded, not GPU-resident)
        self.is_vlm = components['is_vlm']
        self.vision_encoder = components.get('vision_encoder')
        self.projector = components.get('projector')
        if self.is_vlm:
            if self.vision_encoder is not None:
                self.vision_encoder = self.vision_encoder.cpu()
            if self.projector is not None:
                self.projector = self.projector.cpu()
            self._hf_model_type = getattr(hf_model.config, 'model_type', '')
            logger.info(f"VLM mode: vision_encoder + projector on CPU (offloaded)")
        else:
            self._hf_model_type = ''

        # Get config from the text/language model config if available
        cfg = getattr(hf_model.config, 'text_config', hf_model.config)
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        # CPU master modules
        self.embedding = components['embedding'].cpu()
        self.norm = components['norm'].cpu() if components['norm'] else None
        self.lm_head = components['lm_head'].cpu()

        # Detect weight tying (lm_head.weight == embedding.weight)
        self.tied_lm_head = False
        if hasattr(self.lm_head, "weight") and hasattr(self.embedding, "weight"):
            self.tied_lm_head = (self.lm_head.weight is self.embedding.weight)
            if self.tied_lm_head:
                logger.info("Detected tied lm_head and embedding weights")

        # Model-level rotary embedding
        self.rotary_emb = components['rotary_emb'].cpu() if components['rotary_emb'] else None
        if self.rotary_emb:
            logger.info("Found model-level rotary_emb (Qwen2/Llama3/Mistral style)")
        else:
            logger.info("No model-level rotary_emb (layers handle position embeddings internally)")

        # CPU master layers
        self.cpu_layers = [layer.cpu() for layer in components['layers']]

        # === Introspect layer forward signatures ===
        first_layer_params = _introspect_layer_forward(self.cpu_layers[0])
        self.layer_accepts_position_embeddings = 'position_embeddings' in first_layer_params
        self.layer_accepts_position_ids = 'position_ids' in first_layer_params
        self.layer_accepts_cache_position = 'cache_position' in first_layer_params

        logger.info(f"Layer forward accepts: position_embeddings={self.layer_accepts_position_embeddings}, "
                    f"position_ids={self.layer_accepts_position_ids}, "
                    f"cache_position={self.layer_accepts_cache_position}")

        # === Group layers by parameter structure (handles hybrid/MoE) ===
        self.layer_groups, self.layer_to_group = _group_layers_by_structure(self.cpu_layers)

        for gid, group in self.layer_groups.items():
            logger.info(f"Layer group {gid}: {len(group['indices'])} layers, "
                        f"{group['numel'] * config.dtype.itemsize / 1024**2:.1f} MB each "
                        f"(layers: {group['indices'][:5]}{'...' if len(group['indices']) > 5 else ''})")

        # === Per-layer parameter metadata ===
        self.layer_param_shapes = []
        self.layer_param_numel = []
        self.layer_cpu_params = []
        self.layer_numels = []

        for i, layer in enumerate(self.cpu_layers):
            shapes = [p.shape for p in layer.parameters()]
            numel = [p.numel() for p in layer.parameters()]
            cpu_params = list(layer.parameters())
            self.layer_param_shapes.append(shapes)
            self.layer_param_numel.append(numel)
            self.layer_cpu_params.append(cpu_params)
            self.layer_numels.append(sum(numel))

        # Max layer size (for buffer allocation)
        self.max_layer_numel = max(self.layer_numels)
        self.min_layer_numel = min(self.layer_numels)

        if self.max_layer_numel != self.min_layer_numel:
            logger.info(f"Non-uniform layer sizes: min={self.min_layer_numel}, max={self.max_layer_numel} "
                        f"(ratio: {self.max_layer_numel / self.min_layer_numel:.1f}x)")

        # Calculate head (lm_head + norm) and embedding sizes
        self.head_total_numel = sum(p.numel() for p in self.lm_head.parameters())
        if self.norm:
            self.head_total_numel += sum(p.numel() for p in self.norm.parameters())

        self.embed_total_numel = sum(p.numel() for p in self.embedding.parameters())

        # === Pre-flattened pinned buffers (single-GPU only) ===
        self.layer_pinned_flats = []
        if config.world_size == 1:
            for i, layer in enumerate(self.cpu_layers):
                flat = torch.empty(self.layer_numels[i], dtype=config.dtype).pin_memory()
                offset = 0
                for p in layer.parameters():
                    numel = p.numel()
                    flat[offset:offset + numel].copy_(p.data.flatten())
                    offset += numel
                self.layer_pinned_flats.append(flat)

        # Store model config for template creation
        self._model_config = hf_model.config

        # Branch based on world_size
        self.use_multiprocessing = config.world_size > 1
        if self.use_multiprocessing:
            self._init_multiprocessing(config)
        else:
            self._init_single_gpu(config)

        logger.info(f"Model: {len(self.cpu_layers)} layers, checkpoint every {config.checkpoint_interval}")
        logger.info(f"Max flattened param size per layer: {self.max_layer_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"Layer groups: {len(self.layer_groups)}")
        if not self.use_multiprocessing:
            ctx = self.gpu_contexts[0]
            logger.info(f"Gradient slab pools:")
            logger.info(f"  - Layer slabs: {config.num_grad_slabs} x {self.max_layer_numel * config.dtype.itemsize / 1024**2:.2f} MB")
            logger.info(f"  - Head slab: 1 x {self.head_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")
            logger.info(f"  - Embed slab: 1 x {self.embed_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")

    def _init_single_gpu(self, config):
        """Initialize single-GPU mode with a single _GPUContext."""
        ctx = self._create_gpu_context(0, config.devices[0])
        self.gpu_contexts = [ctx]
        self.device = ctx.device

        # CPU worker thread for async gradient accumulation
        self.grad_task_queue = queue.Queue()
        self.worker_stop = threading.Event()
        self.worker_thread = threading.Thread(target=self._grad_worker, daemon=True)
        self.worker_thread.start()

        logger.info(f"Single-GPU mode on cuda:{config.devices[0]}")

    def _init_multiprocessing(self, config):
        """Initialize multi-GPU data parallel mode with worker processes.

        Workers handle forward_logits and forward_and_backward in parallel.
        """
        from infinity.model.mp_state import SharedState
        from infinity.model.mp_worker import gpu_worker_fn

        self.device = torch.device(f"cuda:{config.devices[0]}")

        logger.info(f"Multi-GPU mode: {config.world_size} GPUs on devices {config.devices}")

        # Create shared state (moves params to shared memory)
        self.shared_state = SharedState(self, config)

        # Spawn worker processes
        mp_ctx = mp.get_context('spawn')
        self.worker_processes = []
        for rank in range(config.world_size):
            p = mp_ctx.Process(
                target=gpu_worker_fn,
                args=(rank, self.shared_state),
                daemon=True,
            )
            p.start()
            self.worker_processes.append(p)
            logger.info(f"Spawned worker {rank} (pid={p.pid}) for cuda:{config.devices[rank]}")

        # No local GPU context needed — workers handle all GPU operations.
        self.gpu_contexts = []

    def _create_gpu_context(self, gpu_idx, device_id):
        """Create a _GPUContext with all GPU resources for a single device."""
        device = torch.device(f"cuda:{device_id}")
        ctx = _GPUContext(gpu_idx, device)

        # Double-buffered CPU flat buffers (pinned, sized for max layer)
        ctx.cpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=self.config.dtype).pin_memory(),
            torch.empty(self.max_layer_numel, dtype=self.config.dtype).pin_memory()
        ]

        # Double-buffered GPU flat params
        ctx.gpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=device),
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=device)
        ]

        # GPU layer templates (per structure group, double buffered)
        logger.info("Creating GPU layer templates (per structure group, double buffered)...")
        ctx.gpu_layer_templates = {}
        for gid, group in self.layer_groups.items():
            representative_idx = group['indices'][0]
            templates = []
            for _ in range(2):
                template = copy.deepcopy(self.cpu_layers[representative_idx])
                _preserve_attn_implementation(template, self._model_config)
                template = template.to(device)
                for p in template.parameters():
                    p.requires_grad_(False)
                templates.append(template)
            ctx.gpu_layer_templates[gid] = templates

        # GPU modules (created once, reused)
        logger.info("Creating GPU modules (once)...")
        ctx.emb_gpu = copy.deepcopy(self.embedding).to(device)
        ctx.norm_gpu = copy.deepcopy(self.norm).to(device) if self.norm else None
        ctx.lm_head_gpu = copy.deepcopy(self.lm_head).to(device)

        # Restore weight tying on GPU if detected
        if self.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
            ctx.lm_head_gpu.weight = ctx.emb_gpu.weight
            logger.info("Restored weight tying on GPU (lm_head.weight -> embedding.weight)")

        ctx.rotary_gpu = copy.deepcopy(self.rotary_emb).to(device) if self.rotary_emb else None

        # CUDA streams
        ctx.compute_stream = torch.cuda.current_stream(device=device)
        ctx.weight_stream = torch.cuda.Stream(device=device)
        ctx.grad_stream = torch.cuda.Stream(device=device)

        # Synchronization events
        ctx.weight_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.h2d_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.backward_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.buffer_busy_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.buffer_free_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.template_free_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        ctx.param_sync_event = torch.cuda.Event(enable_timing=False)
        ctx.loss_backward_done = torch.cuda.Event(enable_timing=False)
        ctx.embedding_backward_done = torch.cuda.Event(enable_timing=False)

        # K-slab gradient pool (sized for max layer)
        logger.info(f"Creating categorized gradient slab pools...")
        ctx.layer_grad_slabs = [
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device='cpu', pin_memory=True)
            for _ in range(self.config.num_grad_slabs)
        ]
        ctx.layer_slab_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.config.num_grad_slabs)
        ]
        ctx.layer_slab_free_list = queue.Queue()
        for i in range(self.config.num_grad_slabs):
            ctx.layer_slab_free_list.put(i)

        # Head slab
        ctx.head_grad_slab = torch.empty(self.head_total_numel, dtype=self.config.dtype, device='cpu', pin_memory=True)
        ctx.head_slab_event = torch.cuda.Event(enable_timing=False)
        ctx.head_slab_free = threading.Event()
        ctx.head_slab_free.set()

        # Embedding slab
        ctx.embed_grad_slab = torch.empty(self.embed_total_numel, dtype=self.config.dtype, device='cpu', pin_memory=True)
        ctx.embed_slab_event = torch.cuda.Event(enable_timing=False)
        ctx.embed_slab_free = threading.Event()
        ctx.embed_slab_free.set()

        # Flash-attn CrossEntropyLoss
        if FLASH_CE_AVAILABLE:
            ctx.ce_loss = FlashCrossEntropyLoss(inplace_backward=True, ignore_index=-100, reduction='none')
            logger.info("Using flash-attn CrossEntropyLoss (5-10x less memory)")
        else:
            ctx.ce_loss = None
            logger.info("Flash-attn CE not available, using standard PyTorch CE")

        # Initialize events
        logger.info("Initializing buffer state events...")
        current_stream = torch.cuda.current_stream(device)
        for i in range(2):
            ctx.buffer_free_events[i].record(current_stream)
            ctx.template_free_events[i].record(current_stream)
            ctx.h2d_done_events[i].record(current_stream)
        ctx.param_sync_event.record(current_stream)
        current_stream.synchronize()

        return ctx

    # ------------------------------------------------------------------ #
    #  GPU buffer release / rebuild (single-GPU, VERL integration)
    # ------------------------------------------------------------------ #

    def release_gpu_buffers(self):
        """Release all GPU-resident buffers to free GPU memory.

        Call this when the GPU is needed for other purposes (e.g., inference engine).
        Use rebuild_gpu_buffers() to restore them before training resumes.
        """
        if self.use_multiprocessing:
            self._release_gpu_buffers_multiprocess()
            return

        ctx = self.gpu_contexts[0]
        if not hasattr(self, '_gpu_released') or not self._gpu_released:
            torch.cuda.synchronize(ctx.device)

            # Release double-buffered GPU flat params
            if ctx.gpu_flat_buffers is not None:
                del ctx.gpu_flat_buffers
                ctx.gpu_flat_buffers = None

            # Release GPU layer templates
            if ctx.gpu_layer_templates is not None:
                del ctx.gpu_layer_templates
                ctx.gpu_layer_templates = None

            # Release GPU modules
            if ctx.emb_gpu is not None:
                del ctx.emb_gpu
                ctx.emb_gpu = None
            if ctx.norm_gpu is not None:
                del ctx.norm_gpu
                ctx.norm_gpu = None
            if ctx.lm_head_gpu is not None:
                del ctx.lm_head_gpu
                ctx.lm_head_gpu = None
            if ctx.rotary_gpu is not None:
                del ctx.rotary_gpu
                ctx.rotary_gpu = None

            self._gpu_released = True
            torch.cuda.empty_cache()
            logger.info("Released all GPU buffers from CPUMasterModel")

    def _release_gpu_buffers_multiprocess(self):
        """Release GPU buffers on all workers."""
        from infinity.model.mp_state import WorkerCommand, WorkerCommandType

        # Release all workers
        for rank in range(self.world_size):
            self.shared_state.cmd_queues[rank].put(
                WorkerCommand(type=WorkerCommandType.RELEASE_GPU))
        for rank in range(self.world_size):
            self.shared_state.result_queues[rank].get()

        self._gpu_released = True
        logger.info(f"Released GPU buffers on {self.world_size} workers")

    def rebuild_gpu_buffers(self):
        """Rebuild GPU-resident buffers from CPU state.

        Call this before training resumes after release_gpu_buffers().
        """
        if self.use_multiprocessing:
            self._rebuild_gpu_buffers_multiprocess()
            return

        if not hasattr(self, '_gpu_released') or not self._gpu_released:
            return

        ctx = self.gpu_contexts[0]
        device = ctx.device

        # Rebuild double-buffered GPU flat params
        ctx.gpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=device),
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=device)
        ]

        # Rebuild GPU layer templates
        ctx.gpu_layer_templates = {}
        for gid, group in self.layer_groups.items():
            representative_idx = group['indices'][0]
            templates = []
            for _ in range(2):
                template = copy.deepcopy(self.cpu_layers[representative_idx])
                _preserve_attn_implementation(template, self._model_config)
                template = template.to(device)
                for p in template.parameters():
                    p.requires_grad_(False)
                templates.append(template)
            ctx.gpu_layer_templates[gid] = templates

        # Rebuild GPU modules from CPU state
        ctx.emb_gpu = copy.deepcopy(self.embedding).to(device)
        ctx.norm_gpu = copy.deepcopy(self.norm).to(device) if self.norm else None
        ctx.lm_head_gpu = copy.deepcopy(self.lm_head).to(device)

        if self.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
            ctx.lm_head_gpu.weight = ctx.emb_gpu.weight

        ctx.rotary_gpu = copy.deepcopy(self.rotary_emb).to(device) if self.rotary_emb else None

        # Re-initialize synchronization events
        current_stream = torch.cuda.current_stream(device)
        for i in range(2):
            ctx.buffer_free_events[i].record(current_stream)
            ctx.template_free_events[i].record(current_stream)
            ctx.h2d_done_events[i].record(current_stream)
        ctx.param_sync_event.record(current_stream)
        current_stream.synchronize()

        # Refresh pinned flats from CPU params
        for i, layer in enumerate(self.cpu_layers):
            flat = self.layer_pinned_flats[i]
            offset = 0
            for p in layer.parameters():
                numel = p.numel()
                flat[offset:offset + numel].copy_(p.data.flatten())
                offset += numel

        self._gpu_released = False
        logger.info("Rebuilt all GPU buffers for CPUMasterModel")

    def _rebuild_gpu_buffers_multiprocess(self):
        """Rebuild GPU buffers on all workers."""
        from infinity.model.mp_state import WorkerCommand, WorkerCommandType

        if not hasattr(self, '_gpu_released') or not self._gpu_released:
            return

        # Rebuild all workers
        for rank in range(self.world_size):
            self.shared_state.cmd_queues[rank].put(
                WorkerCommand(type=WorkerCommandType.REBUILD_GPU))
        for rank in range(self.world_size):
            self.shared_state.result_queues[rank].get()

        # Update shared flats so workers have latest weights
        self.shared_state.update_shared_flats()

        self._gpu_released = False
        logger.info(f"Rebuilt GPU buffers on {self.world_size} workers")

    # ------------------------------------------------------------------ #
    #  Internal helpers (single-GPU, ctx-based)
    # ------------------------------------------------------------------ #

    def _get_gpu_layer(self, layer_idx, buffer_idx, ctx):
        """Get the GPU layer template for a given layer index and buffer slot."""
        group_id = self.layer_to_group[layer_idx]
        return ctx.gpu_layer_templates[group_id][buffer_idx]

    def _grad_worker(self):
        """CPU worker thread: wait for D2H completion, accumulate gradients, return slab to pool."""
        while not self.worker_stop.is_set():
            try:
                task = self.grad_task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            slab_type, slab_idx, cpu_params, shapes, numels, ctx = task

            if slab_type == 'layer':
                event = ctx.layer_slab_events[slab_idx]
                slab_flat = ctx.layer_grad_slabs[slab_idx]
            elif slab_type == 'head':
                event = ctx.head_slab_event
                slab_flat = ctx.head_grad_slab
            else:  # 'embed'
                event = ctx.embed_slab_event
                slab_flat = ctx.embed_grad_slab

            event.synchronize()

            offset = 0
            for p_cpu, shape, numel in zip(cpu_params, shapes, numels):
                grad_view = slab_flat[offset:offset + numel].view(shape)
                if p_cpu.grad is None:
                    p_cpu.grad = torch.empty_like(grad_view, device='cpu')
                    p_cpu.grad.copy_(grad_view)
                else:
                    p_cpu.grad.add_(grad_view)
                offset += numel

            if slab_type == 'layer':
                ctx.layer_slab_free_list.put(slab_idx)
            elif slab_type == 'head':
                ctx.head_slab_free.set()
            else:
                ctx.embed_slab_free.set()

            self.grad_task_queue.task_done()

    def _sync_params_to_gpu(self):
        """Sync CPU master params to GPU modules (call after optimizer step)."""
        if self.use_multiprocessing:
            self._sync_params_multiprocess()
            return

        ctx = self.gpu_contexts[0]

        # Refresh pinned flats from updated CPU params
        for i, layer in enumerate(self.cpu_layers):
            flat = self.layer_pinned_flats[i]
            offset = 0
            for p in layer.parameters():
                numel = p.numel()
                flat[offset:offset + numel].copy_(p.data.flatten())
                offset += numel

        # Sync GPU modules
        for p_gpu, p_cpu in zip(ctx.emb_gpu.parameters(), self.embedding.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if ctx.norm_gpu:
            for p_gpu, p_cpu in zip(ctx.norm_gpu.parameters(), self.norm.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if not self.tied_lm_head:
            for p_gpu, p_cpu in zip(ctx.lm_head_gpu.parameters(), self.lm_head.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if ctx.rotary_gpu:
            for p_gpu, p_cpu in zip(ctx.rotary_gpu.parameters(), self.rotary_emb.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        ctx.param_sync_event.record(torch.cuda.current_stream(ctx.device))

    def _sync_params_multiprocess(self):
        """Sync weights to all workers in multi-GPU mode."""
        from infinity.model.mp_state import WorkerCommand, WorkerCommandType

        # Update shared-memory flats from CPU params
        self.shared_state.update_shared_flats()

        # Send SYNC_WEIGHTS to all workers
        for rank in range(self.world_size):
            self.shared_state.cmd_queues[rank].put(
                WorkerCommand(type=WorkerCommandType.SYNC_WEIGHTS)
            )

        # Wait for all workers to finish syncing
        for rank in range(self.world_size):
            self.shared_state.result_queues[rank].get()

    def _load_layer_to_buffer_async(self, layer_idx, buffer_idx, ctx):
        """Load CPU layer params to GPU buffer asynchronously.

        Uses pre-flattened pinned buffers for efficient DMA transfer.
        """
        ctx.h2d_done_events[buffer_idx].synchronize()
        ctx.weight_stream.wait_event(ctx.buffer_free_events[buffer_idx])

        layer_numel = self.layer_numels[layer_idx]

        if self.layer_pinned_flats:
            # Fast path: copy from pre-flattened pinned buffer
            pinned_flat = self.layer_pinned_flats[layer_idx]
            with torch.cuda.stream(ctx.weight_stream):
                ctx.gpu_flat_buffers[buffer_idx][:layer_numel].copy_(
                    pinned_flat, non_blocking=True
                )
                ctx.weight_ready_events[buffer_idx].record(ctx.weight_stream)
                ctx.h2d_done_events[buffer_idx].record(ctx.weight_stream)
        else:
            # Fallback: flatten per-param to CPU staging buffer then copy to GPU
            cpu_flat = ctx.cpu_flat_buffers[buffer_idx]
            layer = self.cpu_layers[layer_idx]

            offset = 0
            for p in layer.parameters():
                numel = p.numel()
                cpu_flat[offset:offset + numel].copy_(p.data.flatten())
                offset += numel

            with torch.cuda.stream(ctx.weight_stream):
                ctx.gpu_flat_buffers[buffer_idx][:layer_numel].copy_(
                    cpu_flat[:layer_numel], non_blocking=True
                )
                ctx.weight_ready_events[buffer_idx].record(ctx.weight_stream)
                ctx.h2d_done_events[buffer_idx].record(ctx.weight_stream)

    def _unflatten_to_layer(self, layer_idx, buffer_idx, ctx):
        """Unflatten GPU buffer to the appropriate layer template parameters."""
        flat = ctx.gpu_flat_buffers[buffer_idx]
        gpu_layer = self._get_gpu_layer(layer_idx, buffer_idx, ctx)

        ctx.compute_stream.wait_event(ctx.template_free_events[buffer_idx])

        offset = 0
        for p in gpu_layer.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)

    def _build_layer_kwargs(self, mask, cache_position, position_ids, position_embeddings):
        """Build kwargs dict for layer forward, based on what the layer accepts."""
        kwargs = {
            'attention_mask': mask,
            'use_cache': False,
            'output_attentions': False,
        }
        if self.layer_accepts_cache_position and cache_position is not None:
            kwargs['cache_position'] = cache_position
        if self.layer_accepts_position_embeddings and position_embeddings is not None:
            kwargs['position_embeddings'] = position_embeddings
        if self.layer_accepts_position_ids and position_ids is not None:
            kwargs['position_ids'] = position_ids
        return kwargs

    def _collect_layer_grads_async(self, layer_idx, buffer_idx, ctx):
        """Collect GPU buffer grads to CPU layer using K-slab flat buffer pool."""
        slab_idx = ctx.layer_slab_free_list.get()
        slab_flat = ctx.layer_grad_slabs[slab_idx]

        ctx.grad_stream.wait_event(ctx.backward_done_events[buffer_idx])

        gpu_layer = self._get_gpu_layer(layer_idx, buffer_idx, ctx)

        with torch.cuda.stream(ctx.grad_stream):
            offset = 0
            for p_gpu in gpu_layer.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad.record_stream(ctx.grad_stream)
                    p_gpu.grad = None
                    offset += numel

            ctx.layer_slab_events[slab_idx].record(ctx.grad_stream)
            ctx.template_free_events[buffer_idx].record(ctx.grad_stream)

        self.grad_task_queue.put((
            'layer',
            slab_idx,
            self.layer_cpu_params[layer_idx],
            self.layer_param_shapes[layer_idx],
            self.layer_param_numel[layer_idx],
            ctx,
        ))

    def _accumulate_grads_batch(self):
        """Wait for CPU worker to finish all gradient accumulation tasks."""
        self.grad_task_queue.join()

    @staticmethod
    def _prepare_4d_causal_mask(mask_2d, dtype, T):
        """Convert 2D padding mask [B, T] to 4D causal attention mask [B, 1, T, T]."""
        B = mask_2d.shape[0]
        device = mask_2d.device

        # Create causal mask
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        # Expand to [B, 1, T, T]
        causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)

        # Create padding mask [B, 1, 1, T]
        padding = (mask_2d == 0).unsqueeze(1).unsqueeze(2)

        # Combine: masked positions get -inf
        combined = causal | padding
        attn_mask = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
        attn_mask.masked_fill_(combined, torch.finfo(dtype).min)

        return attn_mask

    # ------------------------------------------------------------------ #
    #  Vision processing (VLM)
    # ------------------------------------------------------------------ #

    def _process_vision(self, pixel_values, ctx, **vision_kwargs):
        """Process images through vision encoder + projector on GPU, then offload."""
        if self.vision_encoder is None:
            return None

        device = ctx.device

        self.vision_encoder.to(device)
        with torch.no_grad():
            pv = pixel_values.to(device)
            encoder_params = _introspect_layer_forward(self.vision_encoder)
            vkw = {}
            for k, v in vision_kwargs.items():
                val = v.to(device) if isinstance(v, torch.Tensor) else v
                if k in encoder_params:
                    vkw[k] = val
                elif k.startswith("image_") and k[len("image_"):] in encoder_params:
                    vkw[k[len("image_"):]] = val
            if 'grid_thw' in vkw and vkw['grid_thw'].dim() == 3:
                vkw['grid_thw'] = vkw['grid_thw'].reshape(-1, 3)
            try:
                image_features = self.vision_encoder(pv, **vkw)
            except TypeError:
                image_features = self.vision_encoder(pv)
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state
                elif isinstance(image_features, (tuple, list)):
                    image_features = image_features[0]
                elif hasattr(image_features, 'hidden_states'):
                    image_features = image_features.hidden_states
                else:
                    image_features = next(
                        v for v in (image_features.values() if hasattr(image_features, 'values') else [image_features])
                        if isinstance(v, torch.Tensor)
                    )
        self.vision_encoder.cpu()
        torch.cuda.empty_cache()

        if self.projector is not None:
            self.projector.to(device)
            with torch.no_grad():
                image_embeds = self.projector(image_features)
            self.projector.cpu()
            torch.cuda.empty_cache()
        else:
            image_embeds = image_features

        return image_embeds

    def _merge_vision_embeddings(self, hidden, image_embeds, input_ids):
        """Merge image embeddings into text hidden states at image token positions."""
        IMAGE_TOKEN_IDS = set()
        for attr in ['image_token_id', 'vision_start_token_id']:
            tid = getattr(self._model_config, attr, None)
            if tid is not None:
                IMAGE_TOKEN_IDS.add(tid)

        if not IMAGE_TOKEN_IDS:
            n_img_tokens = image_embeds.shape[-2] if image_embeds.dim() == 3 else image_embeds.shape[0]
            for candidate_id in range(151643, 151660):
                mask = (input_ids == candidate_id)
                if mask.sum() > 0:
                    IMAGE_TOKEN_IDS.add(candidate_id)
                    break

        if not IMAGE_TOKEN_IDS:
            logger.warning("Could not find image token positions, prepending image embeddings")
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            return hidden

        merged = hidden.clone()
        img_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in IMAGE_TOKEN_IDS:
            img_mask |= (input_ids == tid)

        if img_mask.sum() > 0 and image_embeds.numel() > 0:
            flat_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            n_positions = img_mask.sum().item()
            n_available = flat_embeds.shape[0]
            n_use = min(n_positions, n_available)
            merged[img_mask][:n_use] = flat_embeds[:n_use]

        return merged

    # ------------------------------------------------------------------ #
    #  Forward-only paths (VERL integration, single-GPU)
    # ------------------------------------------------------------------ #

    def _forward_hidden(self, input_ids, attention_mask, pixel_values=None, **vision_kwargs):
        """Run forward pass through all layers and return final hidden states.

        This is a shared helper used by both inference and training paths.
        Returns (hidden_after_norm, checkpoints, layer_kwargs, input_ids_gpu, B, T).
        """
        ctx = self.gpu_contexts[0]
        B, T = input_ids.shape

        ctx.compute_stream.wait_event(ctx.param_sync_event)

        # VLM: Process images first
        image_embeds = None
        if self.is_vlm and pixel_values is not None:
            image_embeds = self._process_vision(pixel_values, ctx, **vision_kwargs)

        input_ids_gpu = input_ids.to(ctx.device)
        hidden = ctx.emb_gpu(input_ids_gpu)

        if image_embeds is not None:
            hidden = self._merge_vision_embeddings(hidden, image_embeds, input_ids_gpu)
            del image_embeds

        # Position info
        cache_position = torch.arange(T, device=ctx.device)
        position_ids = torch.arange(T, device=ctx.device).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if ctx.rotary_gpu and self.layer_accepts_position_embeddings:
            if self.is_vlm:
                pos_3d = torch.arange(T, device=ctx.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                dummy = torch.empty((1, 1, T, self.head_dim), device=ctx.device, dtype=torch.float32)
                cos, sin = ctx.rotary_gpu(dummy, pos_3d)
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy, pos_3d
            else:
                dummy = torch.empty((1, 1, T, self.head_dim), device=ctx.device, dtype=torch.float32)
                cos, sin = ctx.rotary_gpu(dummy, position_ids[:1])
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy

        mask = attention_mask.to(ctx.device)
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

        checkpoints = {}
        with torch.no_grad():
            self._load_layer_to_buffer_async(0, 0, ctx)
            ctx.weight_stream.synchronize()
            self._unflatten_to_layer(0, 0, ctx)

            for i in range(len(self.cpu_layers)):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = hidden.detach()

                if i + 1 < len(self.cpu_layers):
                    self._load_layer_to_buffer_async(i + 1, next_buffer_idx, ctx)

                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx, ctx)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx, ctx)
                    out = gpu_layer(hidden, **layer_kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out

        checkpoints[len(self.cpu_layers)] = hidden.detach()

        if ctx.norm_gpu:
            hidden_after_norm = ctx.norm_gpu(hidden)
        else:
            hidden_after_norm = hidden

        return hidden_after_norm, checkpoints, layer_kwargs, input_ids_gpu, B, T

    def forward_logits(self, input_ids, attention_mask, pixel_values=None, **vision_kwargs):
        """Forward-only pass that returns logits. Used for inference (rollout, ref policy, etc.).

        In multi-GPU mode, splits the batch across workers for parallel inference.

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: [B, T] attention mask
            pixel_values: Optional image tensor for VLM
            **vision_kwargs: Additional vision kwargs

        Returns:
            logits: [B, T, V] logits tensor on GPU (device 0)
        """
        if self.use_multiprocessing:
            return self._forward_logits_multiprocess(input_ids, attention_mask)

        ctx = self.gpu_contexts[0]
        with torch.no_grad():
            hidden_after_norm, checkpoints, _, _, B, T = self._forward_hidden(
                input_ids, attention_mask, pixel_values, **vision_kwargs
            )
            logits = ctx.lm_head_gpu(hidden_after_norm)
            checkpoints.clear()
            return logits

    def _forward_logits_multiprocess(self, input_ids, attention_mask):
        """Forward-only pass with multi-GPU data parallelism.

        Splits batch across workers, each does full forward independently,
        returns logits on CPU, then concatenates and moves to device 0.
        """
        from infinity.model.mp_state import WorkerCommand, WorkerCommandType

        B = input_ids.shape[0]
        local_bs = B // self.world_size

        # Send commands to workers
        for rank in range(self.world_size):
            start_idx = rank * local_bs
            end_idx = start_idx + local_bs if rank < self.world_size - 1 else B

            cmd = WorkerCommand(
                type=WorkerCommandType.FORWARD_LOGITS,
                input_ids=input_ids[start_idx:end_idx],
                attention_mask=attention_mask[start_idx:end_idx],
            )
            self.shared_state.cmd_queues[rank].put(cmd)

        # Collect results
        all_logits = []
        for rank in range(self.world_size):
            result = self.shared_state.result_queues[rank].get()
            all_logits.append(result.logits)

        # Concatenate on CPU then move to device 0
        logits = torch.cat(all_logits, dim=0).to(self.device)
        return logits

    # ------------------------------------------------------------------ #
    #  Forward + backward
    # ------------------------------------------------------------------ #

    def forward_and_backward(self, input_ids, attention_mask, labels,
                              pixel_values=None, **vision_kwargs):
        """Forward + backward with built-in cross-entropy loss.

        Dispatches to single-GPU or multi-GPU path based on config.
        """
        if self.use_multiprocessing:
            return self._forward_and_backward_multiprocess(
                input_ids, attention_mask, labels, pixel_values, **vision_kwargs
            )

        ctx = self.gpu_contexts[0]
        loss_val, total_tokens, timing, _ = self._forward_and_backward_single_gpu(
            ctx, input_ids, attention_mask, labels, 0,
            pixel_values, **vision_kwargs
        )
        self._accumulate_grads_batch()
        return loss_val, total_tokens, timing

    def _forward_and_backward_single_gpu(self, ctx, input_ids, attention_mask, labels,
                                          global_valid_tokens=0,
                                          pixel_values=None, **vision_kwargs):
        """Forward + backward on a single GPU. Returns (loss_val, total_tokens, timing, valid_tokens)."""
        B, T = input_ids.shape

        ctx.compute_stream.wait_event(ctx.param_sync_event)

        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === FORWARD ===
        image_embeds = None
        if self.is_vlm and pixel_values is not None:
            image_embeds = self._process_vision(pixel_values, ctx, **vision_kwargs)

        input_ids_gpu = input_ids.to(ctx.device)
        hidden = ctx.emb_gpu(input_ids_gpu)

        if image_embeds is not None:
            hidden = self._merge_vision_embeddings(hidden, image_embeds, input_ids_gpu)
            del image_embeds

        del input_ids_gpu

        cache_position = torch.arange(T, device=ctx.device)
        position_ids = torch.arange(T, device=ctx.device).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if ctx.rotary_gpu and self.layer_accepts_position_embeddings:
            if self.is_vlm:
                pos_3d = torch.arange(T, device=ctx.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                dummy = torch.empty((1, 1, T, self.head_dim), device=ctx.device, dtype=torch.float32)
                cos, sin = ctx.rotary_gpu(dummy, pos_3d)
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy, pos_3d
            else:
                dummy = torch.empty((1, 1, T, self.head_dim), device=ctx.device, dtype=torch.float32)
                cos, sin = ctx.rotary_gpu(dummy, position_ids[:1])
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy

        mask = attention_mask.to(ctx.device)
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

        checkpoints = {}

        with torch.no_grad():
            self._load_layer_to_buffer_async(0, 0, ctx)
            ctx.weight_stream.synchronize()
            self._unflatten_to_layer(0, 0, ctx)

            for i in range(len(self.cpu_layers)):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = hidden.detach()

                if i + 1 < len(self.cpu_layers):
                    self._load_layer_to_buffer_async(i + 1, next_buffer_idx, ctx)

                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx, ctx)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx, ctx)
                    out = gpu_layer(hidden, **layer_kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out

        checkpoints[len(self.cpu_layers)] = hidden.detach()

        if ctx.norm_gpu:
            hidden = ctx.norm_gpu(hidden)

        fwd_end.record()

        # === LOSS + BACKWARD ===
        labels_gpu = labels.to(ctx.device)
        V = self.vocab_size
        chunk_size = 128

        hidden_before_norm = checkpoints[len(self.cpu_layers)].requires_grad_(True)
        if ctx.norm_gpu:
            hidden_after_norm = ctx.norm_gpu(hidden_before_norm)
        else:
            hidden_after_norm = hidden_before_norm

        total_loss = torch.zeros((), device=ctx.device, dtype=torch.float32)
        total_valid_tokens = 0

        for t_start in range(0, T - 1, chunk_size):
            t_end = min(t_start + chunk_size, T - 1)
            h = hidden_after_norm[:, t_start:t_end, :]
            y = labels_gpu[:, t_start+1:t_end+1]
            logits = ctx.lm_head_gpu(h)
            flat_y = y.reshape(-1)
            flat_logits = logits.reshape(-1, V)

            if ctx.ce_loss is not None:
                per_tok = ctx.ce_loss(flat_logits, flat_y)
                valid = (flat_y != -100)
                loss_chunk = per_tok[valid].sum()
                total_valid_tokens += int(valid.sum().item())
            else:
                loss_chunk = nn.functional.cross_entropy(
                    flat_logits, flat_y, ignore_index=-100, reduction='sum'
                )
                total_valid_tokens += int((flat_y != -100).sum().item())

            total_loss = total_loss + loss_chunk
            del logits, loss_chunk

        if total_valid_tokens == 0:
            logger.warning("No valid tokens in batch! Skipping...")
            return 0.0, B * T, {'forward': 0.0, 'backward': 0.0, 'total': 0.0}, 0

        denom = global_valid_tokens if global_valid_tokens > 0 else total_valid_tokens
        loss = total_loss / denom
        loss_val = (total_loss / total_valid_tokens).item()

        if not torch.isfinite(torch.tensor(loss_val)):
            logger.error(f"Loss is {loss_val}! Training may be unstable.")

        loss.backward()
        ctx.loss_backward_done.record(ctx.compute_stream)

        grad_hidden = hidden_before_norm.grad.detach()

        # Collect lm_head/norm grads
        if not ctx.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout: worker may be stalled")
        ctx.head_slab_free.clear()
        slab_flat = ctx.head_grad_slab

        with torch.cuda.stream(ctx.grad_stream):
            ctx.grad_stream.wait_event(ctx.loss_backward_done)
            offset = 0
            if not self.tied_lm_head:
                for p_gpu in ctx.lm_head_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            if ctx.norm_gpu:
                for p_gpu in ctx.norm_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            ctx.head_slab_event.record(ctx.grad_stream)

        cpu_params = []
        if not self.tied_lm_head:
            cpu_params.extend(self.lm_head.parameters())
        if self.norm:
            cpu_params.extend(self.norm.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels, ctx))

        del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

        # Backward through layers
        num_blocks = (len(self.cpu_layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval

        for block_idx in range(num_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.cpu_layers))

            current_checkpoint = checkpoints[block_start]

            recompute_cache = {}
            hidden_recompute = current_checkpoint

            with torch.no_grad():
                for j in range(block_start, block_end):
                    buffer_idx = j % 2
                    self._load_layer_to_buffer_async(j, buffer_idx, ctx)
                    ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                    with torch.cuda.stream(ctx.compute_stream):
                        self._unflatten_to_layer(j, buffer_idx, ctx)
                        ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                        gpu_layer = self._get_gpu_layer(j, buffer_idx, ctx)
                        out = gpu_layer(hidden_recompute, **layer_kwargs)
                        hidden_recompute = out[0] if isinstance(out, tuple) else out

                    recompute_cache[j] = hidden_recompute.detach()
                    del out

            for i in range(block_end - 1, block_start - 1, -1):
                buffer_idx = i % 2

                if i == block_start:
                    layer_input = current_checkpoint.detach().requires_grad_(True)
                else:
                    layer_input = recompute_cache[i - 1].requires_grad_(True)

                self._load_layer_to_buffer_async(i, buffer_idx, ctx)
                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx, ctx)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx, ctx)

                    for p in gpu_layer.parameters():
                        p.requires_grad_(True)

                    out = gpu_layer(layer_input, **layer_kwargs)
                    layer_output = out[0] if isinstance(out, tuple) else out

                    grads = torch.autograd.grad(
                        outputs=layer_output,
                        inputs=(layer_input, *gpu_layer.parameters()),
                        grad_outputs=grad_hidden,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_hidden = grads[0].detach()
                    param_grads = grads[1:]

                    for p, g in zip(gpu_layer.parameters(), param_grads):
                        p.grad = g

                    for p in gpu_layer.parameters():
                        p.requires_grad_(False)

                    ctx.backward_done_events[buffer_idx].record(ctx.compute_stream)

                self._collect_layer_grads_async(i, buffer_idx, ctx)

                if i in recompute_cache:
                    del recompute_cache[i]
                del layer_input, layer_output, out

            recompute_cache.clear()

        # === BACKWARD THROUGH EMBEDDING ===
        input_ids_gpu = input_ids.to(ctx.device)
        emb_out = ctx.emb_gpu(input_ids_gpu)

        assert emb_out.shape == grad_hidden.shape, \
            f"Shape mismatch: emb_out {emb_out.shape} vs grad_hidden {grad_hidden.shape}"

        emb_out.backward(grad_hidden)
        ctx.embedding_backward_done.record(ctx.compute_stream)

        if not ctx.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout: worker may be stalled")
        ctx.embed_slab_free.clear()
        slab_flat = ctx.embed_grad_slab

        with torch.cuda.stream(ctx.grad_stream):
            ctx.grad_stream.wait_event(ctx.embedding_backward_done)
            offset = 0
            for p_gpu in ctx.emb_gpu.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad = None
                    offset += numel
            ctx.embed_slab_event.record(ctx.grad_stream)

        cpu_params = list(self.embedding.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels, ctx))

        del input_ids_gpu, emb_out
        del mask, cache_position, position_ids, position_embeddings, grad_hidden
        checkpoints.clear()

        bwd_end.record()
        torch.cuda.synchronize()
        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0
        total_time = start.elapsed_time(bwd_end) / 1000.0

        return loss_val, B * T, {
            'forward': fwd_time,
            'backward': bwd_time,
            'total': total_time,
        }, total_valid_tokens

    def _forward_and_backward_multiprocess(self, input_ids, attention_mask, labels,
                                            pixel_values=None, **vision_kwargs):
        """Forward + backward with multi-GPU data parallelism.

        Splits the batch across workers, each running forward/backward independently.
        Gradients are accumulated to shared-memory tensors.
        """
        from infinity.model.mp_state import WorkerCommand, WorkerCommandType

        B = input_ids.shape[0]
        local_bs = B // self.world_size

        # First pass: count total valid tokens for correct loss normalization
        total_valid_tokens = 0
        for rank in range(self.world_size):
            start_idx = rank * local_bs
            end_idx = start_idx + local_bs
            local_labels = labels[start_idx:end_idx]
            # Valid tokens = labels != -100, shifted by 1
            total_valid_tokens += int((local_labels[:, 1:] != -100).sum().item())

        if total_valid_tokens == 0:
            logger.warning("No valid tokens in entire batch! Skipping...")
            return 0.0, B * input_ids.shape[1], {'forward': 0.0, 'backward': 0.0, 'total': 0.0}

        # Zero shared grads before workers start accumulating
        for p in self.get_parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Send commands to workers
        for rank in range(self.world_size):
            start_idx = rank * local_bs
            end_idx = start_idx + local_bs

            local_pv = None
            if pixel_values is not None:
                local_pv = pixel_values[start_idx:end_idx]

            cmd = WorkerCommand(
                type=WorkerCommandType.FORWARD_BACKWARD,
                input_ids=input_ids[start_idx:end_idx],
                attention_mask=attention_mask[start_idx:end_idx],
                labels=labels[start_idx:end_idx],
                global_valid_tokens=total_valid_tokens,
                pixel_values=local_pv,
            )
            self.shared_state.cmd_queues[rank].put(cmd)

        # Collect results
        results = []
        for rank in range(self.world_size):
            result = self.shared_state.result_queues[rank].get()
            results.append(result)

        # Aggregate loss (weighted by local valid tokens)
        total_loss_sum = sum(r.loss_val * r.valid_tokens for r in results)
        total_valid = sum(r.valid_tokens for r in results)
        avg_loss = total_loss_sum / total_valid if total_valid > 0 else 0.0
        total_tokens = sum(r.total_tokens for r in results)

        # Average timings
        avg_timing = {}
        if results[0].timing:
            for key in results[0].timing:
                avg_timing[key] = sum(r.timing.get(key, 0.0) for r in results) / len(results)

        return avg_loss, total_tokens, avg_timing

    # ------------------------------------------------------------------ #
    #  Forward + backward with custom loss (VERL integration)
    # ------------------------------------------------------------------ #

    def forward_and_backward_custom_loss(self, input_ids, attention_mask, loss_fn,
                                          pixel_values=None, **vision_kwargs):
        """Forward + backward with an externally provided loss function.

        Used by VERL integration where the loss is computed externally (PPO, DPO, etc.)
        rather than using the built-in cross-entropy loss.

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: [B, T] attention mask
            loss_fn: Callable(logits: [B, T, V], input_ids: [B, T]) -> (loss: scalar, meta: dict)
            pixel_values: Optional image tensor for VLM
            **vision_kwargs: Additional vision kwargs

        Returns:
            (loss_val, num_tokens, timing_dict, meta_dict)
        """
        ctx = self.gpu_contexts[0]
        B, T = input_ids.shape

        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        hidden_after_norm, checkpoints, layer_kwargs, input_ids_gpu, B, T = \
            self._forward_hidden(input_ids, attention_mask, pixel_values, **vision_kwargs)

        # Get logits from lm_head (on GPU)
        hidden_before_norm = checkpoints[len(self.cpu_layers)].requires_grad_(True)
        if ctx.norm_gpu:
            hidden_after_norm_grad = ctx.norm_gpu(hidden_before_norm)
        else:
            hidden_after_norm_grad = hidden_before_norm

        logits = ctx.lm_head_gpu(hidden_after_norm_grad)
        fwd_end.record()

        # Call external loss function
        loss, meta = loss_fn(logits, input_ids_gpu)

        if not torch.isfinite(loss):
            logger.error(f"Loss is {loss.item()}! Training may be unstable.")

        loss_val = loss.item()
        loss.backward()
        ctx.loss_backward_done.record(ctx.compute_stream)

        grad_hidden = hidden_before_norm.grad.detach()

        # Collect lm_head/norm grads
        if not ctx.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout: worker may be stalled")
        ctx.head_slab_free.clear()
        slab_flat = ctx.head_grad_slab

        with torch.cuda.stream(ctx.grad_stream):
            ctx.grad_stream.wait_event(ctx.loss_backward_done)
            offset = 0
            if not self.tied_lm_head:
                for p_gpu in ctx.lm_head_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            if ctx.norm_gpu:
                for p_gpu in ctx.norm_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            ctx.head_slab_event.record(ctx.grad_stream)

        cpu_params = []
        if not self.tied_lm_head:
            cpu_params.extend(self.lm_head.parameters())
        if self.norm:
            cpu_params.extend(self.norm.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels, ctx))

        del hidden_after_norm_grad, logits

        # Backward through layers
        num_blocks = (len(self.cpu_layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval

        for block_idx in range(num_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.cpu_layers))

            current_checkpoint = checkpoints[block_start]
            recompute_cache = {}
            hidden_recompute = current_checkpoint

            with torch.no_grad():
                for j in range(block_start, block_end):
                    buffer_idx = j % 2
                    self._load_layer_to_buffer_async(j, buffer_idx, ctx)
                    ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                    with torch.cuda.stream(ctx.compute_stream):
                        self._unflatten_to_layer(j, buffer_idx, ctx)
                        ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                        gpu_layer = self._get_gpu_layer(j, buffer_idx, ctx)
                        out = gpu_layer(hidden_recompute, **layer_kwargs)
                        hidden_recompute = out[0] if isinstance(out, tuple) else out

                    recompute_cache[j] = hidden_recompute.detach()
                    del out

            for i in range(block_end - 1, block_start - 1, -1):
                buffer_idx = i % 2

                if i == block_start:
                    layer_input = current_checkpoint.detach().requires_grad_(True)
                else:
                    layer_input = recompute_cache[i - 1].requires_grad_(True)

                self._load_layer_to_buffer_async(i, buffer_idx, ctx)
                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx, ctx)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx, ctx)

                    for p in gpu_layer.parameters():
                        p.requires_grad_(True)

                    out = gpu_layer(layer_input, **layer_kwargs)
                    layer_output = out[0] if isinstance(out, tuple) else out

                    grads = torch.autograd.grad(
                        outputs=layer_output,
                        inputs=(layer_input, *gpu_layer.parameters()),
                        grad_outputs=grad_hidden,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_hidden = grads[0].detach()
                    param_grads = grads[1:]

                    for p, g in zip(gpu_layer.parameters(), param_grads):
                        p.grad = g

                    for p in gpu_layer.parameters():
                        p.requires_grad_(False)

                    ctx.backward_done_events[buffer_idx].record(ctx.compute_stream)

                self._collect_layer_grads_async(i, buffer_idx, ctx)

                if i in recompute_cache:
                    del recompute_cache[i]
                del layer_input, layer_output, out

            recompute_cache.clear()

        # Backward through embedding
        emb_input = input_ids.to(ctx.device)
        emb_out = ctx.emb_gpu(emb_input)
        emb_out.backward(grad_hidden)
        ctx.embedding_backward_done.record(ctx.compute_stream)

        if not ctx.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout: worker may be stalled")
        ctx.embed_slab_free.clear()
        slab_flat = ctx.embed_grad_slab

        with torch.cuda.stream(ctx.grad_stream):
            ctx.grad_stream.wait_event(ctx.embedding_backward_done)
            offset = 0
            for p_gpu in ctx.emb_gpu.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad = None
                    offset += numel
            ctx.embed_slab_event.record(ctx.grad_stream)

        cpu_params = list(self.embedding.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels, ctx))

        del emb_input, emb_out, grad_hidden
        checkpoints.clear()

        self._accumulate_grads_batch()

        bwd_end.record()
        torch.cuda.synchronize()
        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0

        return loss_val, B * T, {'forward': fwd_time, 'backward': bwd_time}, meta

    # ------------------------------------------------------------------ #
    #  Parameter access / cleanup
    # ------------------------------------------------------------------ #

    def get_parameters(self, include_vision=False):
        """Get all parameters, deduplicated by object id to avoid double-optimizing tied weights."""
        seen = set()
        params = []

        if self.is_vlm and include_vision and self.vision_encoder is not None:
            for p in self.vision_encoder.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        if self.is_vlm and self.projector is not None:
            for p in self.projector.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        for p in self.embedding.parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        for layer in self.cpu_layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        if self.norm is not None:
            for p in self.norm.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        for p in self.lm_head.parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        return params

    def zero_grad(self):
        for p in self.get_parameters():
            if p.grad is not None:
                p.grad.zero_()

    def cleanup(self):
        """Stop worker threads/processes and cleanup resources."""
        if self.use_multiprocessing:
            from infinity.model.mp_state import WorkerCommand, WorkerCommandType
            # Send SHUTDOWN to all workers
            for rank in range(self.world_size):
                self.shared_state.cmd_queues[rank].put(
                    WorkerCommand(type=WorkerCommandType.SHUTDOWN)
                )
            # Wait for workers to exit
            for p in self.worker_processes:
                p.join(timeout=10.0)
                if p.is_alive():
                    logger.warning(f"Worker {p.pid} did not exit, terminating")
                    p.terminate()
            logger.info("All worker processes stopped")
        else:
            self.worker_stop.set()
            self.worker_thread.join(timeout=5.0)
            logger.info("Gradient worker thread stopped")
