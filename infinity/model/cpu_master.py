"""CPU Master Model with explicit recompute and async pipeline.

This module implements a CPU-backed training system for large language models that
exceed GPU memory capacity. Key features:
- FP32 master parameters stored on CPU
- Double-buffered GPU layer execution
- Async weight transfer and gradient collection
- K-slab gradient pool for memory efficiency
- Manual gradient computation without autograd overhead

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


class CPUMasterModel:
    """CPU master with explicit recompute - TRUE async pipeline.

    Supports any HuggingFace decoder-only model and VLM. Handles:
    - Uniform layers (Llama, Qwen2, Mistral, etc.)
    - Hybrid attention (Qwen3.5 linear+full)
    - MoE layers (Mixtral, DeepSeek, Qwen3-Next)
    - VLM (Qwen2-VL, Qwen3-VL, LLaVA, Gemma3-VL, InternVL, etc.)
    """

    def __init__(self, hf_model, config: CPUMasterConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.device}")

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
            # Store the original HF model reference for VLM-specific merge logic
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

        # Model-level rotary embedding (modern HF models: Qwen2, Llama3, Mistral, etc.)
        # Older models (Llama2, GPT-2) compute position embeddings per-layer
        self.rotary_emb = components['rotary_emb'].cpu() if components['rotary_emb'] else None
        if self.rotary_emb:
            logger.info("Found model-level rotary_emb (Qwen2/Llama3/Mistral style)")
        else:
            logger.info("No model-level rotary_emb (layers handle position embeddings internally)")

        # CPU master layers
        self.cpu_layers = [layer.cpu() for layer in components['layers']]

        # === Introspect layer forward signatures ===
        # Different model architectures accept different kwargs
        # We check the first layer of each structure group
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

        # === Double-buffered CPU flat buffers (pinned, sized for max layer) ===
        self.cpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=config.dtype).pin_memory(),
            torch.empty(self.max_layer_numel, dtype=config.dtype).pin_memory()
        ]

        # === Double-buffered GPU flat params ===
        self.gpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device),
            torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device)
        ]

        # === GPU layer templates (per structure group, double buffered) ===
        # CRITICAL: Preserve _attn_implementation when moving layers to GPU.
        # HuggingFace may reset attention implementation during .to(device).
        # We save the config's _attn_implementation and restore it after deepcopy+move.
        self._model_config = hf_model.config
        logger.info("Creating GPU layer templates (per structure group, double buffered)...")
        self.gpu_layer_templates = {}  # {group_id: [template_0, template_1]}
        for gid, group in self.layer_groups.items():
            representative_idx = group['indices'][0]
            templates = []
            for _ in range(2):
                template = copy.deepcopy(self.cpu_layers[representative_idx])
                # Preserve attention implementation before moving to GPU
                _preserve_attn_implementation(template, self._model_config)
                template = template.to(self.device)
                # Ensure no autograd graph is attached to template parameters
                for p in template.parameters():
                    p.requires_grad_(False)
                templates.append(template)
            self.gpu_layer_templates[gid] = templates

        # GPU modules (created once, reused)
        logger.info("Creating GPU modules (once)...")
        self.emb_gpu = copy.deepcopy(self.embedding).to(self.device)
        self.norm_gpu = copy.deepcopy(self.norm).to(self.device) if self.norm else None
        self.lm_head_gpu = copy.deepcopy(self.lm_head).to(self.device)

        # Restore weight tying on GPU if detected
        if self.tied_lm_head and hasattr(self.lm_head_gpu, "weight"):
            self.lm_head_gpu.weight = self.emb_gpu.weight
            logger.info("Restored weight tying on GPU (lm_head.weight -> embedding.weight)")

        self.rotary_gpu = copy.deepcopy(self.rotary_emb).to(self.device) if self.rotary_emb else None

        # === CUDA streams ===
        self.compute_stream = torch.cuda.current_stream(device=self.device)
        self.weight_stream = torch.cuda.Stream(device=self.device)
        self.grad_stream = torch.cuda.Stream(device=self.device)

        # === Synchronization events ===
        self.weight_ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        self.h2d_done_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        self.backward_done_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        self.buffer_busy_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        self.buffer_free_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        # Template protection: template can't be reused until grad D2H finishes
        self.template_free_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(2)
        ]
        self.param_sync_event = torch.cuda.Event(enable_timing=False)
        self.loss_backward_done = torch.cuda.Event(enable_timing=False)
        self.embedding_backward_done = torch.cuda.Event(enable_timing=False)

        # === K-slab gradient pool (sized for max layer) ===
        logger.info(f"Creating categorized gradient slab pools...")

        self.layer_grad_slabs = [
            torch.empty(self.max_layer_numel, dtype=config.dtype, device='cpu', pin_memory=True)
            for _ in range(config.num_grad_slabs)
        ]
        self.layer_slab_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.num_grad_slabs)
        ]
        self.layer_slab_free_list = queue.Queue()
        for i in range(config.num_grad_slabs):
            self.layer_slab_free_list.put(i)

        # Head slab
        self.head_grad_slab = torch.empty(self.head_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        self.head_slab_event = torch.cuda.Event(enable_timing=False)
        self.head_slab_free = threading.Event()
        self.head_slab_free.set()

        # Embedding slab
        self.embed_grad_slab = torch.empty(self.embed_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        self.embed_slab_event = torch.cuda.Event(enable_timing=False)
        self.embed_slab_free = threading.Event()
        self.embed_slab_free.set()

        # CPU worker thread for async gradient accumulation
        self.grad_task_queue = queue.Queue()
        self.worker_stop = threading.Event()
        self.worker_thread = threading.Thread(target=self._grad_worker, daemon=True)
        self.worker_thread.start()

        # Initialize events
        logger.info("Initializing buffer state events...")
        current_stream = torch.cuda.current_stream(self.device)
        for i in range(2):
            self.buffer_free_events[i].record(current_stream)
            self.template_free_events[i].record(current_stream)
            self.h2d_done_events[i].record(current_stream)
        self.param_sync_event.record(current_stream)
        current_stream.synchronize()

        logger.info(f"Model: {len(self.cpu_layers)} layers, checkpoint every {config.checkpoint_interval}")
        logger.info(f"Max flattened param size per layer: {self.max_layer_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"Layer groups: {len(self.layer_groups)}")
        logger.info(f"Gradient slab pools:")
        logger.info(f"  - Layer slabs: {config.num_grad_slabs} x {self.max_layer_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"  - Head slab: 1 x {self.head_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"  - Embed slab: 1 x {self.embed_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")

        # Flash-attn CrossEntropyLoss
        if FLASH_CE_AVAILABLE:
            self.ce_loss = FlashCrossEntropyLoss(inplace_backward=True, ignore_index=-100, reduction='none')
            logger.info("Using flash-attn CrossEntropyLoss (5-10x less memory)")
        else:
            self.ce_loss = None
            logger.info("Flash-attn CE not available, using standard PyTorch CE")

    def release_gpu_buffers(self):
        """Release all GPU-resident buffers to free GPU memory.

        Call this when the GPU is needed for other purposes (e.g., inference engine).
        Use rebuild_gpu_buffers() to restore them before training resumes.
        """
        if not hasattr(self, '_gpu_released') or not self._gpu_released:
            # Synchronize all streams before releasing
            torch.cuda.synchronize(self.device)

            # Release double-buffered GPU flat params
            if hasattr(self, 'gpu_flat_buffers') and self.gpu_flat_buffers is not None:
                del self.gpu_flat_buffers
                self.gpu_flat_buffers = None

            # Release GPU layer templates
            if hasattr(self, 'gpu_layer_templates') and self.gpu_layer_templates is not None:
                del self.gpu_layer_templates
                self.gpu_layer_templates = None

            # Release GPU modules
            if hasattr(self, 'emb_gpu') and self.emb_gpu is not None:
                del self.emb_gpu
                self.emb_gpu = None
            if hasattr(self, 'norm_gpu') and self.norm_gpu is not None:
                del self.norm_gpu
                self.norm_gpu = None
            if hasattr(self, 'lm_head_gpu') and self.lm_head_gpu is not None:
                del self.lm_head_gpu
                self.lm_head_gpu = None
            if hasattr(self, 'rotary_gpu') and self.rotary_gpu is not None:
                del self.rotary_gpu
                self.rotary_gpu = None

            self._gpu_released = True
            torch.cuda.empty_cache()
            logger.info("Released all GPU buffers from CPUMasterModel")

    def rebuild_gpu_buffers(self):
        """Rebuild GPU-resident buffers from CPU state.

        Call this before training resumes after release_gpu_buffers().
        """
        if not hasattr(self, '_gpu_released') or not self._gpu_released:
            return  # Already have GPU buffers

        # Rebuild double-buffered GPU flat params
        self.gpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=self.device),
            torch.empty(self.max_layer_numel, dtype=self.config.dtype, device=self.device)
        ]

        # Rebuild GPU layer templates
        self.gpu_layer_templates = {}
        for gid, group in self.layer_groups.items():
            representative_idx = group['indices'][0]
            templates = []
            for _ in range(2):
                template = copy.deepcopy(self.cpu_layers[representative_idx])
                _preserve_attn_implementation(template, self._model_config)
                template = template.to(self.device)
                for p in template.parameters():
                    p.requires_grad_(False)
                templates.append(template)
            self.gpu_layer_templates[gid] = templates

        # Rebuild GPU modules from CPU state
        self.emb_gpu = copy.deepcopy(self.embedding).to(self.device)
        self.norm_gpu = copy.deepcopy(self.norm).to(self.device) if self.norm else None
        self.lm_head_gpu = copy.deepcopy(self.lm_head).to(self.device)

        if self.tied_lm_head and hasattr(self.lm_head_gpu, "weight"):
            self.lm_head_gpu.weight = self.emb_gpu.weight

        self.rotary_gpu = copy.deepcopy(self.rotary_emb).to(self.device) if self.rotary_emb else None

        # Re-initialize synchronization events
        current_stream = torch.cuda.current_stream(self.device)
        for i in range(2):
            self.buffer_free_events[i].record(current_stream)
            self.template_free_events[i].record(current_stream)
            self.h2d_done_events[i].record(current_stream)
        self.param_sync_event.record(current_stream)
        current_stream.synchronize()

        self._gpu_released = False
        logger.info("Rebuilt all GPU buffers for CPUMasterModel")

    def _get_gpu_layer(self, layer_idx, buffer_idx):
        """Get the GPU layer template for a given layer index and buffer slot."""
        group_id = self.layer_to_group[layer_idx]
        return self.gpu_layer_templates[group_id][buffer_idx]

    def _grad_worker(self):
        """CPU worker thread: wait for D2H completion, accumulate gradients, return slab to pool."""
        while not self.worker_stop.is_set():
            try:
                task = self.grad_task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            slab_type, slab_idx, cpu_params, shapes, numels = task

            if slab_type == 'layer':
                event = self.layer_slab_events[slab_idx]
                slab_flat = self.layer_grad_slabs[slab_idx]
            elif slab_type == 'head':
                event = self.head_slab_event
                slab_flat = self.head_grad_slab
            else:  # 'embed'
                event = self.embed_slab_event
                slab_flat = self.embed_grad_slab

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
                self.layer_slab_free_list.put(slab_idx)
            elif slab_type == 'head':
                self.head_slab_free.set()
            else:
                self.embed_slab_free.set()

            self.grad_task_queue.task_done()

    def _sync_params_to_gpu(self):
        """Sync CPU master params to GPU modules (call after optimizer step)."""
        for p_gpu, p_cpu in zip(self.emb_gpu.parameters(), self.embedding.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if self.norm_gpu:
            for p_gpu, p_cpu in zip(self.norm_gpu.parameters(), self.norm.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if not self.tied_lm_head:
            for p_gpu, p_cpu in zip(self.lm_head_gpu.parameters(), self.lm_head.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if self.rotary_gpu:
            for p_gpu, p_cpu in zip(self.rotary_gpu.parameters(), self.rotary_emb.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        self.param_sync_event.record(torch.cuda.current_stream(self.device))

    def _load_layer_to_buffer_async(self, layer_idx, buffer_idx):
        """Load CPU layer params to GPU buffer asynchronously."""
        self.h2d_done_events[buffer_idx].synchronize()
        self.weight_stream.wait_event(self.buffer_free_events[buffer_idx])

        cpu_flat = self.cpu_flat_buffers[buffer_idx]
        layer = self.cpu_layers[layer_idx]
        layer_numel = self.layer_numels[layer_idx]

        offset = 0
        for p in layer.parameters():
            numel = p.numel()
            cpu_flat[offset:offset + numel].copy_(p.data.flatten())
            offset += numel

        with torch.cuda.stream(self.weight_stream):
            # Only copy the actual layer size, not the full max buffer
            self.gpu_flat_buffers[buffer_idx][:layer_numel].copy_(
                cpu_flat[:layer_numel], non_blocking=True
            )
            self.weight_ready_events[buffer_idx].record(self.weight_stream)
            self.h2d_done_events[buffer_idx].record(self.weight_stream)

    def _unflatten_to_layer(self, layer_idx, buffer_idx):
        """Unflatten GPU buffer to the appropriate layer template parameters.

        After unflatten, the flat buffer is FREE (data copied to template).
        Template is protected by template_free_events until grad D2H completes.
        """
        flat = self.gpu_flat_buffers[buffer_idx]
        gpu_layer = self._get_gpu_layer(layer_idx, buffer_idx)

        # Wait for template to be free (grad D2H from previous use must complete)
        self.compute_stream.wait_event(self.template_free_events[buffer_idx])

        offset = 0
        for p in gpu_layer.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

        # Flat buffer is now free — template holds its own copy
        self.buffer_free_events[buffer_idx].record(self.compute_stream)

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

    def _collect_layer_grads_async(self, layer_idx, buffer_idx):
        """Collect GPU buffer grads to CPU layer using K-slab flat buffer pool."""
        slab_idx = self.layer_slab_free_list.get()
        slab_flat = self.layer_grad_slabs[slab_idx]

        self.grad_stream.wait_event(self.backward_done_events[buffer_idx])

        gpu_layer = self._get_gpu_layer(layer_idx, buffer_idx)

        with torch.cuda.stream(self.grad_stream):
            offset = 0
            for p_gpu in gpu_layer.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad.record_stream(self.grad_stream)
                    p_gpu.grad = None
                    offset += numel

            self.layer_slab_events[slab_idx].record(self.grad_stream)
            # Template is free after grad D2H (NOT buffer_free — flat buffer freed at unflatten)
            self.template_free_events[buffer_idx].record(self.grad_stream)

        self.grad_task_queue.put((
            'layer',
            slab_idx,
            self.layer_cpu_params[layer_idx],
            self.layer_param_shapes[layer_idx],
            self.layer_param_numel[layer_idx]
        ))

    def _accumulate_grads_batch(self):
        """Wait for CPU worker to finish all gradient accumulation tasks."""
        self.grad_task_queue.join()

    def _process_vision(self, pixel_values, **vision_kwargs):
        """Process images through vision encoder + projector on GPU, then offload.

        Both vision encoder and projector are loaded to GPU on-demand and
        offloaded back to CPU after use, keeping GPU memory free for decoder layers.

        Args:
            pixel_values: Image tensor from processor
            **vision_kwargs: Additional kwargs (image_grid_thw, etc.)

        Returns:
            image_embeds: Projected image embeddings on GPU [N_images, N_tokens, hidden_size]
        """
        if self.vision_encoder is None:
            return None

        # 1. Load vision encoder to GPU, process images
        self.vision_encoder.to(self.device)
        with torch.no_grad():
            pv = pixel_values.to(self.device)
            # Map processor output keys to vision encoder parameter names.
            # HF processors output "image_grid_thw" but Qwen-VL vision encoders
            # expect "grid_thw". Introspect the encoder's forward signature and
            # strip the common "image_" prefix when the encoder doesn't accept
            # the prefixed name but does accept the short name.
            encoder_params = _introspect_layer_forward(self.vision_encoder)
            vkw = {}
            for k, v in vision_kwargs.items():
                val = v.to(self.device) if isinstance(v, torch.Tensor) else v
                if k in encoder_params:
                    vkw[k] = val
                elif k.startswith("image_") and k[len("image_"):] in encoder_params:
                    vkw[k[len("image_"):]] = val
                # else: skip keys the encoder doesn't accept
            # grid_thw may have an extra batch dim from collation [B, N_img, 3];
            # vision encoders expect [total_images, 3], so flatten if needed.
            if 'grid_thw' in vkw and vkw['grid_thw'].dim() == 3:
                vkw['grid_thw'] = vkw['grid_thw'].reshape(-1, 3)
            try:
                image_features = self.vision_encoder(pv, **vkw)
            except TypeError:
                # Fallback: some encoders don't accept extra kwargs
                image_features = self.vision_encoder(pv)
            # Handle non-tensor output (ModelOutput, tuple, etc.)
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state
                elif isinstance(image_features, (tuple, list)):
                    image_features = image_features[0]
                elif hasattr(image_features, 'hidden_states'):
                    image_features = image_features.hidden_states
                else:
                    # Generic fallback: grab first tensor value
                    image_features = next(
                        v for v in (image_features.values() if hasattr(image_features, 'values') else [image_features])
                        if isinstance(v, torch.Tensor)
                    )
        self.vision_encoder.cpu()
        torch.cuda.empty_cache()
        logger.debug(f"Vision encoder done, features shape: {image_features.shape}")

        # 2. Load projector to GPU, project features
        if self.projector is not None:
            self.projector.to(self.device)
            with torch.no_grad():
                image_embeds = self.projector(image_features)
            self.projector.cpu()
            torch.cuda.empty_cache()
        else:
            image_embeds = image_features

        logger.debug(f"Projector done, embeds shape: {image_embeds.shape}")
        return image_embeds

    def _merge_vision_embeddings(self, hidden, image_embeds, input_ids):
        """Merge image embeddings into text hidden states at image token positions.

        Replaces hidden states at positions where input_ids match the image token
        with the projected image embeddings.

        Args:
            hidden: Text embeddings [B, T, H]
            image_embeds: Image embeddings [N_img_tokens, H] or [B, N_img_tokens, H]
            input_ids: Input token IDs [B, T] (on GPU)

        Returns:
            Merged hidden states [B, T, H]
        """
        # Find image token positions
        # Common image token IDs across models
        IMAGE_TOKEN_IDS = set()
        # Try to get from the HF model config (not the training config)
        for attr in ['image_token_id', 'vision_start_token_id']:
            tid = getattr(self._model_config, attr, None)
            if tid is not None:
                IMAGE_TOKEN_IDS.add(tid)

        if not IMAGE_TOKEN_IDS:
            # Fallback: scan for token ID that appears repeatedly and matches image embed count
            # This is a heuristic; specific models may need specific handling
            n_img_tokens = image_embeds.shape[-2] if image_embeds.dim() == 3 else image_embeds.shape[0]
            for candidate_id in range(151643, 151660):  # Qwen-VL image token range
                mask = (input_ids == candidate_id)
                if mask.sum() > 0:
                    IMAGE_TOKEN_IDS.add(candidate_id)
                    break

        if not IMAGE_TOKEN_IDS:
            # If we can't find image tokens, prepend image embeddings
            logger.warning("Could not find image token positions, prepending image embeddings")
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            # This is a simplistic fallback; real models handle this in their own forward
            return hidden

        # Replace image token positions with image embeddings
        merged = hidden.clone()
        img_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in IMAGE_TOKEN_IDS:
            img_mask |= (input_ids == tid)

        # Flatten and replace
        if img_mask.sum() > 0 and image_embeds.numel() > 0:
            flat_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            n_positions = img_mask.sum().item()
            n_available = flat_embeds.shape[0]
            n_use = min(n_positions, n_available)
            merged[img_mask][:n_use] = flat_embeds[:n_use]

        return merged

    def _forward_hidden(self, input_ids, attention_mask, pixel_values=None, **vision_kwargs):
        """Run forward pass through all layers and return final hidden states.

        This is a shared helper used by both inference and training paths.
        Returns (hidden_after_norm, hidden_before_norm, layer_kwargs, checkpoints, B, T).
        """
        B, T = input_ids.shape

        self.compute_stream.wait_event(self.param_sync_event)

        # === VLM: Process images first ===
        image_embeds = None
        if self.is_vlm and pixel_values is not None:
            image_embeds = self._process_vision(pixel_values, **vision_kwargs)

        input_ids_gpu = input_ids.to(self.device)
        hidden = self.emb_gpu(input_ids_gpu)

        if image_embeds is not None:
            hidden = self._merge_vision_embeddings(hidden, image_embeds, input_ids_gpu)
            del image_embeds

        # Position info
        cache_position = torch.arange(T, device=self.device)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if self.rotary_gpu and self.layer_accepts_position_embeddings:
            if self.is_vlm:
                pos_3d = torch.arange(T, device=self.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                dummy = torch.empty((1, 1, T, self.head_dim), device=self.device, dtype=torch.float32)
                cos, sin = self.rotary_gpu(dummy, pos_3d)
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy, pos_3d
            else:
                dummy = torch.empty((1, 1, T, self.head_dim), device=self.device, dtype=torch.float32)
                cos, sin = self.rotary_gpu(dummy, position_ids[:1])
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy

        mask = attention_mask.to(self.device)
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

        checkpoints = {}
        with torch.no_grad():
            self._load_layer_to_buffer_async(0, 0)
            self.weight_stream.synchronize()
            self._unflatten_to_layer(0, 0)

            for i in range(len(self.cpu_layers)):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = hidden.detach()

                if i + 1 < len(self.cpu_layers):
                    self._load_layer_to_buffer_async(i + 1, next_buffer_idx)

                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx)
                    out = gpu_layer(hidden, **layer_kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out

        checkpoints[len(self.cpu_layers)] = hidden.detach()

        if self.norm_gpu:
            hidden_after_norm = self.norm_gpu(hidden)
        else:
            hidden_after_norm = hidden

        return hidden_after_norm, checkpoints, layer_kwargs, input_ids_gpu, B, T

    def forward_logits(self, input_ids, attention_mask, pixel_values=None, **vision_kwargs):
        """Forward-only pass that returns logits. Used for inference (rollout, ref policy, etc.).

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: [B, T] attention mask
            pixel_values: Optional image tensor for VLM
            **vision_kwargs: Additional vision kwargs

        Returns:
            logits: [B, T, V] logits tensor on GPU
        """
        with torch.no_grad():
            hidden_after_norm, checkpoints, _, _, B, T = self._forward_hidden(
                input_ids, attention_mask, pixel_values, **vision_kwargs
            )
            logits = self.lm_head_gpu(hidden_after_norm)
            checkpoints.clear()
            return logits

    def forward_and_backward(self, input_ids, attention_mask, labels,
                              pixel_values=None, **vision_kwargs):
        B, T = input_ids.shape

        self.compute_stream.wait_event(self.param_sync_event)

        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === FORWARD ===

        # === VLM: Process images first (vision encoder + projector on GPU, then offload) ===
        image_embeds = None
        if self.is_vlm and pixel_values is not None:
            image_embeds = self._process_vision(pixel_values, **vision_kwargs)

        input_ids_gpu = input_ids.to(self.device)
        hidden = self.emb_gpu(input_ids_gpu)

        # Merge image embeddings into hidden states at image token positions
        if image_embeds is not None:
            hidden = self._merge_vision_embeddings(hidden, image_embeds, input_ids_gpu)
            del image_embeds

        del input_ids_gpu

        # Position info (model-agnostic)
        cache_position = torch.arange(T, device=self.device)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)

        # Compute position_embeddings if model has model-level rotary_emb
        position_embeddings = None
        if self.rotary_gpu and self.layer_accepts_position_embeddings:
            if self.is_vlm:
                # VLM models (e.g., Qwen2.5-VL) use M-RoPE with 3D position_ids [3, B, T]
                # For text-only input, use simple sequential positions for all 3 dims
                pos_3d = torch.arange(T, device=self.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                dummy = torch.empty((1, 1, T, self.head_dim), device=self.device, dtype=torch.float32)
                cos, sin = self.rotary_gpu(dummy, pos_3d)
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy, pos_3d
            else:
                dummy = torch.empty((1, 1, T, self.head_dim), device=self.device, dtype=torch.float32)
                cos, sin = self.rotary_gpu(dummy, position_ids[:1])
                position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
                del dummy

        # Attention mask: pass as-is (2D), let HF layers handle 4D expansion
        mask = attention_mask.to(self.device)

        # Build layer kwargs once
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

        # Checkpoints
        checkpoints = {}

        with torch.no_grad():
            self._load_layer_to_buffer_async(0, 0)
            self.weight_stream.synchronize()
            self._unflatten_to_layer(0, 0)

            for i in range(len(self.cpu_layers)):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = hidden.detach()

                if i + 1 < len(self.cpu_layers):
                    self._load_layer_to_buffer_async(i + 1, next_buffer_idx)

                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx)
                    out = gpu_layer(hidden, **layer_kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out

        checkpoints[len(self.cpu_layers)] = hidden.detach()

        if self.norm_gpu:
            hidden = self.norm_gpu(hidden)

        fwd_end.record()

        # === LOSS + BACKWARD ===
        labels_gpu = labels.to(self.device)
        H = self.hidden_size
        V = self.vocab_size
        chunk_size = 128

        hidden_before_norm = checkpoints[len(self.cpu_layers)].requires_grad_(True)
        if self.norm_gpu:
            hidden_after_norm = self.norm_gpu(hidden_before_norm)
        else:
            hidden_after_norm = hidden_before_norm

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_valid_tokens = 0

        for t_start in range(0, T - 1, chunk_size):
            t_end = min(t_start + chunk_size, T - 1)
            h = hidden_after_norm[:, t_start:t_end, :]
            y = labels_gpu[:, t_start+1:t_end+1]
            logits = self.lm_head_gpu(h)
            flat_y = y.reshape(-1)
            flat_logits = logits.reshape(-1, V)

            if self.ce_loss is not None:
                per_tok = self.ce_loss(flat_logits, flat_y)
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
            return 0.0, B * T, {'forward': 0.0, 'backward': 0.0}

        loss = total_loss / total_valid_tokens
        loss_val = loss.item()

        if not torch.isfinite(torch.tensor(loss_val)):
            logger.error(f"Loss is {loss_val}! Training may be unstable.")

        loss.backward()
        self.loss_backward_done.record(self.compute_stream)

        grad_hidden = hidden_before_norm.grad.detach()

        # Collect lm_head/norm grads
        if not self.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout: worker may be stalled")
        self.head_slab_free.clear()
        slab_flat = self.head_grad_slab

        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.loss_backward_done)
            offset = 0
            if not self.tied_lm_head:
                for p_gpu in self.lm_head_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            if self.norm_gpu:
                for p_gpu in self.norm_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            self.head_slab_event.record(self.grad_stream)

        cpu_params = []
        if not self.tied_lm_head:
            cpu_params.extend(self.lm_head.parameters())
        if self.norm_gpu:
            cpu_params.extend(self.norm.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels))

        del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

        # Backward through layers
        num_blocks = (len(self.cpu_layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval

        for block_idx in range(num_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.cpu_layers))

            current_checkpoint = checkpoints[block_start]

            # Recompute block
            recompute_cache = {}
            hidden_recompute = current_checkpoint

            with torch.no_grad():
                for j in range(block_start, block_end):
                    buffer_idx = j % 2
                    self._load_layer_to_buffer_async(j, buffer_idx)
                    self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                    with torch.cuda.stream(self.compute_stream):
                        self._unflatten_to_layer(j, buffer_idx)
                        self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                        gpu_layer = self._get_gpu_layer(j, buffer_idx)
                        out = gpu_layer(hidden_recompute, **layer_kwargs)
                        hidden_recompute = out[0] if isinstance(out, tuple) else out

                    recompute_cache[j] = hidden_recompute.detach()
                    del out

            # Backward through block
            for i in range(block_end - 1, block_start - 1, -1):
                buffer_idx = i % 2

                if i == block_start:
                    layer_input = current_checkpoint.detach().requires_grad_(True)
                else:
                    layer_input = recompute_cache[i - 1].requires_grad_(True)

                self._load_layer_to_buffer_async(i, buffer_idx)
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx)

                    # Temporarily enable gradients on GPU template parameters for autograd.grad
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

                    # Disable gradients again to keep templates clean
                    for p in gpu_layer.parameters():
                        p.requires_grad_(False)

                    self.backward_done_events[buffer_idx].record(self.compute_stream)

                self._collect_layer_grads_async(i, buffer_idx)

                if i in recompute_cache:
                    del recompute_cache[i]
                del layer_input, layer_output, out

            recompute_cache.clear()

        # === BACKWARD THROUGH EMBEDDING ===
        input_ids_gpu = input_ids.to(self.device)
        emb_out = self.emb_gpu(input_ids_gpu)

        assert emb_out.shape == grad_hidden.shape, \
            f"Shape mismatch: emb_out {emb_out.shape} vs grad_hidden {grad_hidden.shape}"

        emb_out.backward(grad_hidden)
        self.embedding_backward_done.record(self.compute_stream)

        if not self.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout: worker may be stalled")
        self.embed_slab_free.clear()
        slab_flat = self.embed_grad_slab

        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.embedding_backward_done)
            offset = 0
            for p_gpu in self.emb_gpu.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad = None
                    offset += numel
            self.embed_slab_event.record(self.grad_stream)

        cpu_params = list(self.embedding.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels))

        del input_ids_gpu, emb_out
        del mask, cache_position, position_ids, position_embeddings, grad_hidden
        checkpoints.clear()

        self._accumulate_grads_batch()

        bwd_end.record()
        torch.cuda.synchronize()
        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0
        total_time = start.elapsed_time(bwd_end) / 1000.0

        total_tokens_for_throughput = B * T

        return loss_val, total_tokens_for_throughput, {
            'forward': fwd_time,
            'backward': bwd_time,
            'total': total_time,
        }

    def forward_and_backward_custom_loss(self, input_ids, attention_mask, loss_fn,
                                          pixel_values=None, **vision_kwargs):
        """Forward + backward with an externally provided loss function.

        Used by VERL integration where the loss is computed externally (PPO, DPO, etc.)
        rather than using the built-in cross-entropy loss.

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: [B, T] attention mask
            loss_fn: Callable(logits: [B, T, V], input_ids: [B, T]) -> (loss: scalar, meta: dict)
                     The loss function receives full logits and input_ids, and returns
                     a scalar loss for backprop and a metadata dict.
            pixel_values: Optional image tensor for VLM
            **vision_kwargs: Additional vision kwargs

        Returns:
            (loss_val, num_tokens, timing_dict, meta_dict)
        """
        B, T = input_ids.shape

        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        hidden_after_norm, checkpoints, layer_kwargs, input_ids_gpu, B, T = \
            self._forward_hidden(input_ids, attention_mask, pixel_values, **vision_kwargs)

        # Get logits from lm_head (on GPU)
        hidden_before_norm = checkpoints[len(self.cpu_layers)].requires_grad_(True)
        if self.norm_gpu:
            hidden_after_norm_grad = self.norm_gpu(hidden_before_norm)
        else:
            hidden_after_norm_grad = hidden_before_norm

        logits = self.lm_head_gpu(hidden_after_norm_grad)
        fwd_end.record()

        # Call external loss function
        loss, meta = loss_fn(logits, input_ids_gpu)

        if not torch.isfinite(loss):
            logger.error(f"Loss is {loss.item()}! Training may be unstable.")

        loss_val = loss.item()
        loss.backward()
        self.loss_backward_done.record(self.compute_stream)

        grad_hidden = hidden_before_norm.grad.detach()

        # Collect lm_head/norm grads
        if not self.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout: worker may be stalled")
        self.head_slab_free.clear()
        slab_flat = self.head_grad_slab

        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.loss_backward_done)
            offset = 0
            if not self.tied_lm_head:
                for p_gpu in self.lm_head_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            if self.norm_gpu:
                for p_gpu in self.norm_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel
            self.head_slab_event.record(self.grad_stream)

        cpu_params = []
        if not self.tied_lm_head:
            cpu_params.extend(self.lm_head.parameters())
        if self.norm_gpu:
            cpu_params.extend(self.norm.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels))

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
                    self._load_layer_to_buffer_async(j, buffer_idx)
                    self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                    with torch.cuda.stream(self.compute_stream):
                        self._unflatten_to_layer(j, buffer_idx)
                        self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                        gpu_layer = self._get_gpu_layer(j, buffer_idx)
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

                self._load_layer_to_buffer_async(i, buffer_idx)
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(i, buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    gpu_layer = self._get_gpu_layer(i, buffer_idx)

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

                    self.backward_done_events[buffer_idx].record(self.compute_stream)

                self._collect_layer_grads_async(i, buffer_idx)

                if i in recompute_cache:
                    del recompute_cache[i]
                del layer_input, layer_output, out

            recompute_cache.clear()

        # Backward through embedding
        emb_input = input_ids.to(self.device)
        emb_out = self.emb_gpu(emb_input)
        emb_out.backward(grad_hidden)
        self.embedding_backward_done.record(self.compute_stream)

        if not self.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout: worker may be stalled")
        self.embed_slab_free.clear()
        slab_flat = self.embed_grad_slab

        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.embedding_backward_done)
            offset = 0
            for p_gpu in self.emb_gpu.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad = None
                    offset += numel
            self.embed_slab_event.record(self.grad_stream)

        cpu_params = list(self.embedding.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels))

        del emb_input, emb_out, grad_hidden
        checkpoints.clear()

        self._accumulate_grads_batch()

        bwd_end.record()
        torch.cuda.synchronize()
        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0

        return loss_val, B * T, {'forward': fwd_time, 'backward': bwd_time}, meta

    def get_parameters(self, include_vision=False):
        """Get all parameters, deduplicated by object id to avoid double-optimizing tied weights.

        Args:
            include_vision: If True, include vision encoder parameters (default: False,
                           as vision encoder is typically frozen during VLM fine-tuning)
        """
        seen = set()
        params = []

        # VLM: optionally include vision encoder and projector
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
        """Stop worker thread and cleanup resources."""
        self.worker_stop.set()
        self.worker_thread.join(timeout=5.0)
