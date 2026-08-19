"""Tests for CPUMasterModel.compute_loss (forward-only evaluation loss).

The guard tests run on CPU. The numerical tests need a GPU, because
CPUMasterModel allocates pinned host memory, CUDA streams and device-resident
buffers in its constructor and has no CPU fallback.

Run standalone:
    pytest tests/test_compute_loss.py -v
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CPUMasterModel requires CUDA"
)


def _build_tiny_model():
    """Build a small real Llama and wrap it in a CPUMasterModel.

    Returns (hf_model, cpu_master). Caller must call cpu_master.cleanup().
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from infinity.config.training import CPUMasterConfig
    from infinity.model.cpu_master import CPUMasterModel

    torch.manual_seed(0)
    hf_config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    hf_model = LlamaForCausalLM(hf_config)

    config = CPUMasterConfig(
        model_name="tiny-llama-for-tests",
        dataset_name="dummy",          # __post_init__ requires a dataset
        attn_implementation="eager",   # no flash-attn dependency
        dtype=torch.float32,           # tight tolerance for the parity check
        max_seq_len=32,
        batch_size=2,
        checkpoint_interval=2,
        num_grad_slabs=4,
    )
    return hf_model, CPUMasterModel(hf_model, config)


def _make_batch(B=2, T=16, vocab_size=256, mask_prefix=4):
    """Build a batch where the first mask_prefix positions are ignored."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :mask_prefix] = -100
    return input_ids, attention_mask, labels


# --------------------------------------------------------------------- #
#  CPU-only: multi-GPU guard
# --------------------------------------------------------------------- #

def test_compute_loss_rejects_multi_gpu():
    """compute_loss must refuse multi-GPU rather than silently using one GPU.

    Constructed via __new__ to skip the CUDA-dependent __init__: the guard is
    the first statement in the method, so no GPU state is needed to reach it.
    """
    from infinity.model.cpu_master import CPUMasterModel

    model = CPUMasterModel.__new__(CPUMasterModel)
    model.use_multiprocessing = True
    model.world_size = 4
    model.gpu_contexts = []  # what _init_multiprocessing leaves behind

    with pytest.raises(NotImplementedError) as excinfo:
        model.compute_loss(
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
        )

    message = str(excinfo.value)
    assert "world_size=4" in message, f"error should report world_size: {message}"
    assert "single-GPU" in message, f"error should name the supported mode: {message}"

    # The guard must precede `ctx = self.gpu_contexts[0]`; if it did not, an
    # empty gpu_contexts would surface as a bare IndexError instead.
    assert not isinstance(excinfo.value, IndexError)
    print("[ok] compute_loss rejects multi-GPU with an actionable message")


def test_multi_gpu_guard_is_a_runtime_error():
    """Callers catching RuntimeError should also catch this guard."""
    assert issubclass(NotImplementedError, RuntimeError)
    print("[ok] NotImplementedError is catchable as RuntimeError")


# --------------------------------------------------------------------- #
#  GPU: numerical behaviour
# --------------------------------------------------------------------- #

@requires_cuda
def test_compute_loss_matches_training_loss():
    """Eval loss must equal the training path's loss for the same batch."""
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, labels = _make_batch()

        # Evaluate first, so no backward has run yet.
        eval_loss, eval_tokens = model.compute_loss(input_ids, attention_mask, labels)

        train_loss, _, _ = model.forward_and_backward(input_ids, attention_mask, labels)

        expected_tokens = int((labels[:, 1:] != -100).sum().item())
        assert eval_tokens == expected_tokens, (
            f"valid token count {eval_tokens} != expected {expected_tokens}"
        )

        assert eval_loss == pytest.approx(train_loss, rel=1e-4), (
            f"eval loss {eval_loss} != train loss {train_loss}"
        )
        print(f"[ok] eval loss {eval_loss:.6f} matches train loss {train_loss:.6f}")
    finally:
        model.cleanup()


@requires_cuda
def test_compute_loss_produces_no_gradients():
    """compute_loss must not populate a gradient on any parameter."""
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, labels = _make_batch()

        model.zero_grad()
        for p in model.get_parameters():
            p.grad = None

        model.compute_loss(input_ids, attention_mask, labels)

        with_grad = [i for i, p in enumerate(model.get_parameters()) if p.grad is not None]
        assert not with_grad, f"CPU master params gained gradients at indices {with_grad}"

        ctx = model.gpu_contexts[0]
        gpu_modules = {"emb_gpu": ctx.emb_gpu, "lm_head_gpu": ctx.lm_head_gpu}
        if ctx.norm_gpu is not None:
            gpu_modules["norm_gpu"] = ctx.norm_gpu
        for name, module in gpu_modules.items():
            for p in module.parameters():
                assert p.grad is None, f"{name} gained a gradient"

        print("[ok] compute_loss leaves every parameter gradient unset")
    finally:
        model.cleanup()


@requires_cuda
def test_compute_loss_does_not_modify_weights():
    """Evaluation must be side-effect free on the master weights."""
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, labels = _make_batch()

        before = [p.detach().clone() for p in model.get_parameters()]
        model.compute_loss(input_ids, attention_mask, labels)

        for i, (p, snapshot) in enumerate(zip(model.get_parameters(), before)):
            assert torch.equal(p.detach(), snapshot), f"parameter {i} was modified"

        print("[ok] compute_loss leaves master weights unchanged")
    finally:
        model.cleanup()


@requires_cuda
def test_compute_loss_fully_masked_batch():
    """A batch with no supervised positions returns (0.0, 0), not NaN."""
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, _ = _make_batch()
        labels = torch.full_like(input_ids, -100)

        loss, tokens = model.compute_loss(input_ids, attention_mask, labels)

        assert loss == 0.0, f"expected 0.0 loss for a fully masked batch, got {loss}"
        assert tokens == 0, f"expected 0 valid tokens, got {tokens}"
        print("[ok] fully masked batch returns (0.0, 0)")
    finally:
        model.cleanup()


@requires_cuda
def test_compute_loss_is_repeatable():
    """Repeated evaluation of the same batch gives the same loss."""
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, labels = _make_batch()

        first = model.compute_loss(input_ids, attention_mask, labels)
        second = model.compute_loss(input_ids, attention_mask, labels)

        assert first[1] == second[1]
        assert first[0] == pytest.approx(second[0], rel=1e-6)
        print(f"[ok] compute_loss is repeatable ({first[0]:.6f})")
    finally:
        model.cleanup()


@requires_cuda
def test_forward_hidden_keeps_final_checkpoint_without_recompute_anchors():
    """Skipping recompute anchors must not drop the final hidden state.

    forward_and_backward_custom_loss reads checkpoints[len(cpu_layers)], so that
    entry has to survive collect_recompute_checkpoints=False.
    """
    hf_model, model = _build_tiny_model()
    try:
        input_ids, attention_mask, _ = _make_batch()
        num_layers = len(model.cpu_layers)

        with torch.no_grad():
            _, full, _, _, _, _ = model._forward_hidden(
                input_ids, attention_mask, None, collect_recompute_checkpoints=True
            )
            full_keys = set(full.keys())
            full.clear()

            _, lean, _, _, _, _ = model._forward_hidden(
                input_ids, attention_mask, None, collect_recompute_checkpoints=False
            )
            lean_keys = set(lean.keys())
            lean.clear()

        assert num_layers in full_keys, "default path must keep the final hidden state"
        assert lean_keys == {num_layers}, (
            f"forward-only path should keep only the final entry, got {sorted(lean_keys)}"
        )
        assert len(full_keys) > len(lean_keys), (
            "default path should collect more anchors than the forward-only path"
        )
        print(f"[ok] recompute anchors {sorted(full_keys)} -> {sorted(lean_keys)}")
    finally:
        model.cleanup()
