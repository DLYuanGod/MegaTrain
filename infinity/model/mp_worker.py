"""Worker process for multiprocessing data parallelism.

Each worker process owns one GPU and runs forward/backward independently.
Workers read shared-memory weight buffers for H2D and write gradients to
shared-memory accumulators with a lock.
"""

import logging
import copy
import threading
import queue
import torch
import torch.nn as nn

from infinity.model.cpu_master import (
    _GPUContext, _preserve_attn_implementation, CPUMasterModel,
    FLASH_CE_AVAILABLE,
)
if FLASH_CE_AVAILABLE:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss

from infinity.model.mp_state import WorkerCommandType, WorkerResult

logger = logging.getLogger(__name__)


def _create_worker_gpu_context(rank, device_id, shared_state):
    """Create a _GPUContext for this worker's GPU.

    Similar to CPUMasterModel._create_gpu_context but reads from
    shared_state instead of self.
    """
    config = shared_state.config
    device = torch.device(f"cuda:{device_id}")
    ctx = _GPUContext(rank, device)

    logger.info(f"Worker {rank}: creating GPU context on {device}...")

    # Double-buffered GPU flat buffers (no CPU pinned staging needed —
    # we copy directly from shared-memory flats to GPU)
    ctx.gpu_flat_buffers = [
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device=device),
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device=device)
    ]
    # Unused but kept for API compat with _unflatten_to_layer
    ctx.cpu_flat_buffers = [None, None]

    # GPU layer templates (per structure group, double buffered)
    ctx.gpu_layer_templates = {}
    for gid, group in shared_state.layer_groups.items():
        representative_idx = group['indices'][0]
        templates = []
        for _ in range(2):
            template = copy.deepcopy(shared_state.cpu_layers[representative_idx])
            _preserve_attn_implementation(template, shared_state._model_config)
            template = template.to(device)
            for p in template.parameters():
                p.requires_grad_(False)
            templates.append(template)
        ctx.gpu_layer_templates[gid] = templates

    # GPU modules
    ctx.emb_gpu = copy.deepcopy(shared_state.embedding).to(device)
    ctx.norm_gpu = copy.deepcopy(shared_state.norm).to(device) if shared_state.norm else None
    ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(device)

    if shared_state.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
        ctx.lm_head_gpu.weight = ctx.emb_gpu.weight

    ctx.rotary_gpu = copy.deepcopy(shared_state.rotary_emb).to(device) if shared_state.rotary_emb else None

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

    # K-slab gradient pool (per-worker, pinned CPU memory)
    ctx.layer_grad_slabs = [
        torch.empty(shared_state.max_layer_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        for _ in range(config.num_grad_slabs)
    ]
    ctx.layer_slab_events = [
        torch.cuda.Event(enable_timing=False) for _ in range(config.num_grad_slabs)
    ]
    ctx.layer_slab_free_list = queue.Queue()
    for i in range(config.num_grad_slabs):
        ctx.layer_slab_free_list.put(i)

    ctx.head_grad_slab = torch.empty(shared_state.head_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
    ctx.head_slab_event = torch.cuda.Event(enable_timing=False)
    ctx.head_slab_free = threading.Event()
    ctx.head_slab_free.set()

    ctx.embed_grad_slab = torch.empty(shared_state.embed_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
    ctx.embed_slab_event = torch.cuda.Event(enable_timing=False)
    ctx.embed_slab_free = threading.Event()
    ctx.embed_slab_free.set()

    # Flash CE loss
    if FLASH_CE_AVAILABLE:
        ctx.ce_loss = FlashCrossEntropyLoss(inplace_backward=True, ignore_index=-100, reduction='none')
    else:
        ctx.ce_loss = None

    # Initialize events
    current_stream = torch.cuda.current_stream(device)
    for i in range(2):
        ctx.buffer_free_events[i].record(current_stream)
        ctx.template_free_events[i].record(current_stream)
        ctx.h2d_done_events[i].record(current_stream)
    ctx.param_sync_event.record(current_stream)
    current_stream.synchronize()

    return ctx


def _worker_grad_fn(grad_queue, worker_stop, shared_state, ctx):
    """Per-worker gradient accumulation thread.

    Same as CPUMasterModel._grad_worker but writes to shared-memory
    p_cpu.grad tensors with a lock for cross-process safety.
    """
    while not worker_stop.is_set():
        try:
            task = grad_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        slab_type, slab_idx, cpu_params, shapes, numels = task

        if slab_type == 'layer':
            event = ctx.layer_slab_events[slab_idx]
            slab_flat = ctx.layer_grad_slabs[slab_idx]
        elif slab_type == 'head':
            event = ctx.head_slab_event
            slab_flat = ctx.head_grad_slab
        else:
            event = ctx.embed_slab_event
            slab_flat = ctx.embed_grad_slab

        event.synchronize()

        # Accumulate to shared-memory p_cpu.grad with lock
        with shared_state.grad_lock:
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

        grad_queue.task_done()


def _worker_load_layer_to_buffer_async(layer_idx, buffer_idx, ctx, shared_state):
    """Load layer weights from shared-memory flat to GPU buffer.

    Direct shared-memory → GPU copy (non_blocking). Not as fast as
    pinned → GPU DMA, but avoids the need for pinned shared memory.
    """
    ctx.h2d_done_events[buffer_idx].synchronize()
    ctx.weight_stream.wait_event(ctx.buffer_free_events[buffer_idx])

    shared_flat = shared_state.layer_shared_flats[layer_idx]
    layer_numel = shared_state.layer_numels[layer_idx]

    with torch.cuda.stream(ctx.weight_stream):
        ctx.gpu_flat_buffers[buffer_idx][:layer_numel].copy_(
            shared_flat, non_blocking=True
        )
        ctx.weight_ready_events[buffer_idx].record(ctx.weight_stream)
        ctx.h2d_done_events[buffer_idx].record(ctx.weight_stream)


def _worker_sync_gpu_modules(ctx, shared_state):
    """Refresh worker's GPU modules from shared CPU master params."""
    for p_gpu, p_cpu in zip(ctx.emb_gpu.parameters(), shared_state.embedding.parameters()):
        p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if ctx.norm_gpu:
        for p_gpu, p_cpu in zip(ctx.norm_gpu.parameters(), shared_state.norm.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if not shared_state.tied_lm_head:
        for p_gpu, p_cpu in zip(ctx.lm_head_gpu.parameters(), shared_state.lm_head.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    if ctx.rotary_gpu:
        for p_gpu, p_cpu in zip(ctx.rotary_gpu.parameters(), shared_state.rotary_emb.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

    ctx.param_sync_event.record(torch.cuda.current_stream(ctx.device))


def gpu_worker_fn(rank, shared_state):
    """Main entry point for a worker process (spawned via mp.spawn).

    Creates a GPU context, starts a gradient worker thread, and enters
    a command loop receiving work from the main process.
    """
    device_id = shared_state.device_ids[rank]
    torch.cuda.set_device(device_id)

    # Re-attach shared-memory grad tensors lost during spawn pickling
    shared_state.reattach_grads()

    # Register shared-memory flats with CUDA for pinned DMA speed
    shared_state.register_shared_flats_as_pinned()

    logger.info(f"Worker {rank}: starting on cuda:{device_id}")

    ctx = _create_worker_gpu_context(rank, device_id, shared_state)

    # Per-process grad worker thread
    grad_queue = queue.Queue()
    worker_stop = threading.Event()
    grad_thread = threading.Thread(
        target=_worker_grad_fn,
        args=(grad_queue, worker_stop, shared_state, ctx),
        daemon=True,
    )
    grad_thread.start()

    cmd_q = shared_state.cmd_queues[rank]
    result_q = shared_state.result_queues[rank]

    while True:
        cmd = cmd_q.get()

        if cmd.type == WorkerCommandType.SHUTDOWN:
            logger.info(f"Worker {rank}: shutting down")
            break

        elif cmd.type == WorkerCommandType.SYNC_WEIGHTS:
            _worker_sync_gpu_modules(ctx, shared_state)
            result_q.put(WorkerResult())

        elif cmd.type == WorkerCommandType.FORWARD_BACKWARD:
            result = _run_forward_backward(
                rank, ctx, shared_state, grad_queue, cmd)
            # Wait for all grad tasks to complete before signaling main
            grad_queue.join()
            result_q.put(result)

    worker_stop.set()
    grad_thread.join(timeout=5.0)
    logger.info(f"Worker {rank}: exited")


def _run_forward_backward(rank, ctx, shared_state, grad_queue, cmd):
    """Run forward + backward on this worker's GPU.

    Adapted from CPUMasterModel._forward_and_backward_single_gpu.
    Uses shared_state for weight loading and grad_queue for gradient collection.
    """
    input_ids = cmd.input_ids
    attention_mask = cmd.attention_mask
    labels = cmd.labels
    global_valid_tokens = cmd.global_valid_tokens
    pixel_values = cmd.pixel_values

    B, T = input_ids.shape

    ctx.compute_stream.wait_event(ctx.param_sync_event)

    start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    start.record(ctx.compute_stream)

    # === FORWARD ===
    input_ids_gpu = input_ids.to(ctx.device)
    hidden = ctx.emb_gpu(input_ids_gpu)
    del input_ids_gpu

    cache_position = torch.arange(T, device=ctx.device)
    position_ids = torch.arange(T, device=ctx.device).unsqueeze(0).expand(B, -1)

    position_embeddings = None
    if ctx.rotary_gpu and shared_state.layer_accepts_position_embeddings:
        if shared_state.is_vlm:
            pos_3d = torch.arange(T, device=ctx.device).unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            dummy = torch.empty((1, 1, T, shared_state.head_dim), device=ctx.device, dtype=torch.float32)
            cos, sin = ctx.rotary_gpu(dummy, pos_3d)
            position_embeddings = (cos.to(shared_state.config.dtype), sin.to(shared_state.config.dtype))
            del dummy, pos_3d
        else:
            dummy = torch.empty((1, 1, T, shared_state.head_dim), device=ctx.device, dtype=torch.float32)
            cos, sin = ctx.rotary_gpu(dummy, position_ids[:1])
            position_embeddings = (cos.to(shared_state.config.dtype), sin.to(shared_state.config.dtype))
            del dummy

    mask = attention_mask.to(ctx.device)
    # Reuse the static method from CPUMasterModel
    if mask is not None and mask.dim() == 2 and shared_state.config.attn_implementation != 'flash_attention_2':
        mask = CPUMasterModel._prepare_4d_causal_mask(mask, shared_state.config.dtype, T)

    layer_kwargs = {
        'attention_mask': mask,
        'use_cache': False,
        'output_attentions': False,
    }
    if shared_state.layer_accepts_cache_position and cache_position is not None:
        layer_kwargs['cache_position'] = cache_position
    if shared_state.layer_accepts_position_embeddings and position_embeddings is not None:
        layer_kwargs['position_embeddings'] = position_embeddings
    if shared_state.layer_accepts_position_ids and position_ids is not None:
        layer_kwargs['position_ids'] = position_ids

    checkpoints = {}
    num_layers = len(shared_state.cpu_layers)
    checkpoint_interval = shared_state.config.checkpoint_interval

    with torch.no_grad():
        _worker_load_layer_to_buffer_async(0, 0, ctx, shared_state)
        ctx.weight_stream.synchronize()
        _worker_unflatten_to_layer(0, 0, ctx, shared_state)

        for i in range(num_layers):
            buffer_idx = i % 2
            next_buffer_idx = (i + 1) % 2

            if i % checkpoint_interval == 0:
                checkpoints[i] = hidden.detach()

            if i + 1 < num_layers:
                _worker_load_layer_to_buffer_async(i + 1, next_buffer_idx, ctx, shared_state)

            ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

            with torch.cuda.stream(ctx.compute_stream):
                _worker_unflatten_to_layer(i, buffer_idx, ctx, shared_state)
                ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                group_id = shared_state.layer_to_group[i]
                gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]
                out = gpu_layer(hidden, **layer_kwargs)
                hidden = out[0] if isinstance(out, tuple) else out

    checkpoints[num_layers] = hidden.detach()

    if ctx.norm_gpu:
        hidden = ctx.norm_gpu(hidden)

    fwd_end.record(ctx.compute_stream)

    # === LOSS + BACKWARD ===
    labels_gpu = labels.to(ctx.device)
    V = shared_state.vocab_size
    chunk_size = 128

    hidden_before_norm = checkpoints[num_layers].requires_grad_(True)
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
        return WorkerResult(loss_val=0.0, total_tokens=B * T, valid_tokens=0)

    denom = global_valid_tokens if global_valid_tokens > 0 else total_valid_tokens
    loss = total_loss / denom
    loss_val = (total_loss / total_valid_tokens).item()

    if not torch.isfinite(torch.tensor(loss_val)):
        logger.error(f"Worker {rank}: Loss is {loss_val}!")

    loss.backward()
    ctx.loss_backward_done.record(ctx.compute_stream)

    grad_hidden = hidden_before_norm.grad.detach()

    # Collect lm_head/norm grads
    if not ctx.head_slab_free.wait(timeout=30.0):
        raise RuntimeError("head slab wait timeout")
    ctx.head_slab_free.clear()
    slab_flat = ctx.head_grad_slab

    with torch.cuda.stream(ctx.grad_stream):
        ctx.grad_stream.wait_event(ctx.loss_backward_done)
        offset = 0
        if not shared_state.tied_lm_head:
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
    if not shared_state.tied_lm_head:
        cpu_params.extend(shared_state.lm_head.parameters())
    if shared_state.norm:
        cpu_params.extend(shared_state.norm.parameters())
    shapes = [p.shape for p in cpu_params]
    numels = [p.numel() for p in cpu_params]
    grad_queue.put(('head', None, cpu_params, shapes, numels))

    del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

    # Backward through layers
    num_blocks = (num_layers + checkpoint_interval - 1) // checkpoint_interval

    for block_idx in range(num_blocks - 1, -1, -1):
        block_start = block_idx * checkpoint_interval
        block_end = min((block_idx + 1) * checkpoint_interval, num_layers)

        current_checkpoint = checkpoints[block_start]

        recompute_cache = {}
        hidden_recompute = current_checkpoint

        with torch.no_grad():
            for j in range(block_start, block_end):
                buffer_idx = j % 2
                _worker_load_layer_to_buffer_async(j, buffer_idx, ctx, shared_state)
                ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

                with torch.cuda.stream(ctx.compute_stream):
                    _worker_unflatten_to_layer(j, buffer_idx, ctx, shared_state)
                    ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                    group_id = shared_state.layer_to_group[j]
                    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]
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

            _worker_load_layer_to_buffer_async(i, buffer_idx, ctx, shared_state)
            ctx.compute_stream.wait_event(ctx.weight_ready_events[buffer_idx])

            with torch.cuda.stream(ctx.compute_stream):
                _worker_unflatten_to_layer(i, buffer_idx, ctx, shared_state)
                ctx.buffer_busy_events[buffer_idx].record(ctx.compute_stream)

                group_id = shared_state.layer_to_group[i]
                gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

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

            _worker_collect_layer_grads_async(i, buffer_idx, ctx, shared_state, grad_queue)

            if i in recompute_cache:
                del recompute_cache[i]
            del layer_input, layer_output, out

        recompute_cache.clear()

    # === BACKWARD THROUGH EMBEDDING ===
    input_ids_gpu = input_ids.to(ctx.device)
    emb_out = ctx.emb_gpu(input_ids_gpu)
    emb_out.backward(grad_hidden)
    ctx.embedding_backward_done.record(ctx.compute_stream)

    if not ctx.embed_slab_free.wait(timeout=30.0):
        raise RuntimeError("embed slab wait timeout")
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

    cpu_params = list(shared_state.embedding.parameters())
    shapes = [p.shape for p in cpu_params]
    numels = [p.numel() for p in cpu_params]
    grad_queue.put(('embed', None, cpu_params, shapes, numels))

    del input_ids_gpu, emb_out
    del mask, cache_position, position_ids, position_embeddings, grad_hidden
    checkpoints.clear()

    bwd_end.record(ctx.compute_stream)
    torch.cuda.synchronize(ctx.device)
    fwd_time = start.elapsed_time(fwd_end) / 1000.0
    bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0

    return WorkerResult(
        loss_val=loss_val,
        total_tokens=B * T,
        timing={'forward': fwd_time, 'backward': bwd_time, 'total': fwd_time + bwd_time},
        valid_tokens=total_valid_tokens,
    )


def _worker_unflatten_to_layer(layer_idx, buffer_idx, ctx, shared_state):
    """Unflatten GPU buffer to the appropriate layer template parameters."""
    flat = ctx.gpu_flat_buffers[buffer_idx]
    group_id = shared_state.layer_to_group[layer_idx]
    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

    ctx.compute_stream.wait_event(ctx.template_free_events[buffer_idx])

    offset = 0
    for p in gpu_layer.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view(p.shape))
        offset += numel

    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)


def _worker_collect_layer_grads_async(layer_idx, buffer_idx, ctx, shared_state, grad_queue):
    """Collect GPU buffer grads to CPU using K-slab pool."""
    slab_idx = ctx.layer_slab_free_list.get()
    slab_flat = ctx.layer_grad_slabs[slab_idx]

    ctx.grad_stream.wait_event(ctx.backward_done_events[buffer_idx])

    group_id = shared_state.layer_to_group[layer_idx]
    gpu_layer = ctx.gpu_layer_templates[group_id][buffer_idx]

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

    grad_queue.put((
        'layer',
        slab_idx,
        shared_state.layer_cpu_params[layer_idx],
        shared_state.layer_param_shapes[layer_idx],
        shared_state.layer_param_numel[layer_idx],
    ))
