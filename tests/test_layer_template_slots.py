"""Exercise template allocation and reuse with small, real hybrid models.

Models are randomly initialized locally; no weights or tokenizer are downloaded.
GLM's first dense layer only uses slot 0. Qwen's full-attention layers only use
slot 1. Their other structure groups use both slots.
"""

import gc

import pytest
import torch

from infinity.config.training import CPUMasterConfig
from infinity.model.cpu_master import (
    CPUMasterModel,
    _discover_model_components,
    _group_layers_by_structure,
)
from infinity.model.mp_worker import (
    _create_worker_gpu_context,
    _worker_rebuild_gpu,
    _worker_release_gpu,
    _worker_unflatten_to_layer,
)


def tiny_model(architecture, dtype=torch.float32):
    """Keep the real layer types and ordering, with inexpensive dimensions."""
    torch.manual_seed(42)
    common = dict(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    if architecture == "glm":
        module = pytest.importorskip(
            "transformers.models.glm4_moe.modeling_glm4_moe"
        )
        config = module.Glm4MoeConfig(
            **common, num_hidden_layers=4, intermediate_size=128,
            moe_intermediate_size=64, n_routed_experts=4,
            n_shared_experts=1, num_experts_per_tok=2,
            n_group=1, topk_group=1, first_k_dense_replace=1,
        )
        model_class = module.Glm4MoeForCausalLM
    else:
        module = pytest.importorskip(
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
        )
        config = module.Qwen3_5MoeTextConfig(
            **common, num_hidden_layers=8, head_dim=16,
            layer_types=["linear_attention"] * 3 + ["full_attention"]
            + ["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            moe_intermediate_size=64, shared_expert_intermediate_size=64,
            num_experts=4, num_experts_per_tok=2,
            rope_parameters={
                "rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3],
            },
        )
        model_class = module.Qwen3_5MoeForCausalLM
    return model_class._from_config(
        config, attn_implementation="sdpa", dtype=dtype,
    ).eval()


def make_master(architecture, *, world_size=1, interval=2, dtype=torch.float32):
    config = CPUMasterConfig(
        dataset_name="local-test", devices=list(range(world_size)),
        batch_size=2, max_seq_len=16, dtype=dtype,
        attn_implementation="sdpa", checkpoint_interval=interval,
        num_grad_slabs=2 * interval,
    )
    return CPUMasterModel(tiny_model(architecture, dtype), config)


@pytest.mark.parametrize("architecture", ["glm", "qwen"])
def test_real_models_cover_all_slot_patterns(architecture):
    layers = _discover_model_components(tiny_model(architecture))["layers"]
    groups, _ = _group_layers_by_structure(layers)
    actual = [group["indices"] for group in groups.values()]
    expected = [[0], [1, 2, 3]] if architecture == "glm" else [
        [0, 1, 2, 4, 5, 6], [3, 7],
    ]
    assert actual == expected


def assert_templates(master, ctx, *, worker):
    """Check allocation, independent storage, and actual unflatten selection."""
    for gid, group in master.layer_groups.items():
        templates = ctx.gpu_layer_templates[gid]
        used = {i % 2 for i in group["indices"]}
        assert len(templates) == 2
        assert {i for i, layer in enumerate(templates) if layer is not None} == used
        if len(used) == 2:
            for p0, p1 in zip(templates[0].parameters(), templates[1].parameters()):
                assert p0.data_ptr() != p1.data_ptr()

    for index, cpu_layer in enumerate(master.cpu_layers):
        slot = index % 2
        flat = torch.cat([p.detach().reshape(-1) for p in cpu_layer.parameters()])
        ctx.gpu_flat_buffers[slot][:flat.numel()].copy_(flat)
        if worker:
            _worker_unflatten_to_layer(index, slot, ctx, master)
        else:
            master._unflatten_to_layer(index, slot, ctx)
        layer = ctx.gpu_layer_templates[master.layer_to_group[index]][slot]
        for actual, expected in zip(layer.parameters(), cpu_layer.parameters()):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("architecture", ["glm", "qwen"])
@pytest.mark.parametrize("worker", [False, True], ids=["single", "worker"])
def test_template_allocation_and_rebuild(architecture, worker):
    master = make_master(architecture)
    ctx = None
    try:
        # Worker allocation consumes the same CPU modules and metadata as the
        # single-GPU master, so it can be exercised without starting a process.
        ctx = (_create_worker_gpu_context(0, 0, master) if worker
               else master.gpu_contexts[0])
        assert_templates(master, ctx, worker=worker)
        torch.cuda.synchronize()
        if worker:
            _worker_release_gpu(ctx)
            assert ctx.gpu_layer_templates == {}
            _worker_rebuild_gpu(ctx, master)
        else:
            master.release_gpu_buffers()
            assert ctx.gpu_layer_templates is None
            master.rebuild_gpu_buffers()
        assert_templates(master, ctx, worker=worker)
    finally:
        torch.cuda.synchronize()
        if worker and ctx is not None:
            _worker_release_gpu(ctx)
        master.cleanup()
        master.release_gpu_buffers()
        del ctx, master
        gc.collect()
        torch.cuda.empty_cache()


def training_snapshots(architecture, world_size, interval, dtype, rebuild):
    """Two updates, optionally rebuilding between them, with a fixed batch."""
    master = make_master(
        architecture, world_size=world_size, interval=interval, dtype=dtype,
    )
    generator = torch.Generator().manual_seed(17)
    input_ids = torch.randint(0, 128, (2, 16), generator=generator)
    labels = input_ids.clone()
    labels[:, :3] = -100
    # The single-GPU SDPA entry point expects a prepared causal mask.
    mask = CPUMasterModel._prepare_4d_causal_mask(
        torch.ones_like(input_ids), dtype, input_ids.shape[1],
    )
    parameters = master.get_parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    snapshots = []
    try:
        for step in range(2):
            master.zero_grad()
            logits = master.forward_logits(input_ids, mask).detach().cpu().clone()
            loss, _, _ = master.forward_and_backward(input_ids, mask, labels)
            gradients = [
                None if p.grad is None else p.grad.detach().cpu().clone()
                for p in parameters
            ]
            assert torch.isfinite(logits).all()
            assert torch.isfinite(torch.tensor(loss))
            assert any(g is not None and g.count_nonzero() for g in gradients)
            assert all(g is None or torch.isfinite(g).all() for g in gradients)
            optimizer.step()
            master._sync_params_to_gpu()
            snapshots.append({
                "logits": logits, "loss": torch.tensor(loss),
                "gradients": gradients,
                "parameters": [p.detach().cpu().clone() for p in parameters],
            })
            if rebuild and step == 0:
                master.release_gpu_buffers()
                master.rebuild_gpu_buffers()
        return snapshots
    finally:
        master.cleanup()
        if not master.use_multiprocessing:
            master.release_gpu_buffers()
        del optimizer, parameters, master
        gc.collect()
        torch.cuda.empty_cache()


def assert_snapshots_close(actual, expected, dtype):
    tolerances = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {}
    assert len(actual) == len(expected)
    for step, (left, right) in enumerate(zip(actual, expected)):
        for key in ("logits", "loss", "gradients", "parameters"):
            x, y = left[key], right[key]
            if key in ("gradients", "parameters"):
                assert len(x) == len(y)
                for index, (a, b) in enumerate(zip(x, y)):
                    assert (a is None) == (b is None), (step, key, index)
                    if a is not None:
                        torch.testing.assert_close(a, b, **tolerances)
            else:
                torch.testing.assert_close(x, y, **tolerances)


@pytest.mark.parametrize("architecture", ["glm", "qwen"])
@pytest.mark.parametrize("world_size", [1, 2], ids=["single", "dp2"])
@pytest.mark.parametrize("interval", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_training_survives_rebuild(architecture, world_size, interval, dtype):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    continuous = training_snapshots(architecture, world_size, interval, dtype, False)
    rebuilt = training_snapshots(architecture, world_size, interval, dtype, True)
    assert_snapshots_close(rebuilt, continuous, dtype)
