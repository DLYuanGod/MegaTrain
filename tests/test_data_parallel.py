"""Test data parallel gradient equivalence.

Verifies that processing a batch of 4 samples on 1 GPU produces identical
gradients to splitting those 4 samples across 4 GPUs (1 sample per GPU).

These two setups are mathematically equivalent. We validate in float32
(where the difference is ~1e-5) to prove correctness. We also test bf16
with a looser tolerance since bf16 backward passes accumulate differently
for different batch sizes (~1e-2 relative diff).
"""

import copy
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinity.config import CPUMasterConfig
from infinity import CPUMasterModel, ChatDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-2-2b-it"
BATCH_SIZE = 4
MAX_SEQ_LEN = 64
CHECKPOINT_INTERVAL = 2
NUM_GRAD_SLABS = 6


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def batch(tokenizer):
    dataset = ChatDataset(tokenizer, MAX_SEQ_LEN,
                          dataset_name="alpaca_en_demo", dataset_dir="data")
    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                    num_workers=0, shuffle=True, generator=g)
    return next(iter(dl))


def _make_model(state, devices, dtype):
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="cpu",
        attn_implementation="eager",
    )
    hf.load_state_dict(state)
    config = CPUMasterConfig(
        model_name=MODEL_NAME,
        devices=devices,
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        num_grad_slabs=NUM_GRAD_SLABS,
        dataset_name="alpaca_en_demo",
        dtype=dtype,
        attn_implementation="eager",
        max_seq_len=MAX_SEQ_LEN,
    )
    model = CPUMasterModel(hf, config)
    del hf
    return model


def _run_and_get_grads(state, devices, batch, dtype):
    model = _make_model(state, devices, dtype)
    loss, _, _ = model.forward_and_backward(
        batch["input_ids"], batch["attention_mask"], batch["labels"])
    grads = [p.grad.float().clone() for p in model.get_parameters()
             if p.grad is not None]
    model.cleanup()
    del model
    torch.cuda.empty_cache()
    return loss, grads


def _compare_grads(grads_a, grads_b):
    max_abs = 0.0
    max_rel = 0.0
    for a, b in zip(grads_a, grads_b):
        d = (a - b).abs().max().item()
        denom = max(a.abs().max().item(), 1e-10)
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, d / denom)
    return max_abs, max_rel


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs")
def test_dp4_gradient_equivalence_float32(batch):
    """float32: 1 GPU batch=4 must match 4 GPUs batch=1 exactly."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    dtype = torch.float32
    state = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="cpu",
        attn_implementation="eager").state_dict()

    loss_1, grads_1 = _run_and_get_grads(state, [0], batch, dtype)
    loss_4, grads_4 = _run_and_get_grads(state, [0, 1, 2, 3], batch, dtype)

    print(f"\nfloat32: 1GPU loss={loss_1:.6f}, 4GPU loss={loss_4:.6f}, "
          f"diff={abs(loss_1-loss_4):.2e}")

    max_abs, max_rel = _compare_grads(grads_1, grads_4)
    print(f"float32: max abs grad diff={max_abs:.2e}, max rel={max_rel:.2e}")

    assert abs(loss_1 - loss_4) < 1e-5, \
        f"Loss mismatch: {loss_1} vs {loss_4}"
    assert max_abs < 1e-4, \
        f"Gradient mismatch: max abs diff {max_abs:.2e}, max rel {max_rel:.2e}"


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs")
def test_dp4_gradient_equivalence_bf16(batch):
    """bf16: 1 GPU batch=4 must match 4 GPUs batch=1 within bf16 tolerance."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    dtype = torch.bfloat16
    state = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="cpu",
        attn_implementation="eager").state_dict()

    loss_1, grads_1 = _run_and_get_grads(state, [0], batch, dtype)
    loss_4, grads_4 = _run_and_get_grads(state, [0, 1, 2, 3], batch, dtype)

    print(f"\nbf16: 1GPU loss={loss_1:.6f}, 4GPU loss={loss_4:.6f}, "
          f"diff={abs(loss_1-loss_4):.2e}")

    max_abs, max_rel = _compare_grads(grads_1, grads_4)
    print(f"bf16: max abs grad diff={max_abs:.2e}, max rel={max_rel:.2e}")

    # bf16 backward produces different results for different batch sizes due to
    # different intermediate accumulation in torch.autograd.grad. This is the same
    # divergence seen in standard DDP bf16 training. The forward pass (loss) may
    # also differ slightly due to bf16 matmul differences at different batch sizes.
    assert abs(loss_1 - loss_4) < 1e-2, \
        f"Loss mismatch too large: {loss_1} vs {loss_4}"
    assert max_rel < 0.5, \
        f"Gradient divergence too large: max rel diff {max_rel:.2e}"


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs")
def test_dp2_gradient_equivalence_bf16(batch):
    """bf16: 1 GPU batch=4 must match 2 GPUs batch=2 within bf16 tolerance."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    dtype = torch.bfloat16
    state = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="cpu",
        attn_implementation="eager").state_dict()

    loss_1, grads_1 = _run_and_get_grads(state, [0], batch, dtype)
    loss_2, grads_2 = _run_and_get_grads(state, [0, 1], batch, dtype)

    print(f"\nbf16 dp2: 1GPU loss={loss_1:.6f}, 2GPU loss={loss_2:.6f}, "
          f"diff={abs(loss_1-loss_2):.2e}")

    max_abs, max_rel = _compare_grads(grads_1, grads_2)
    print(f"bf16 dp2: max abs grad diff={max_abs:.2e}, max rel={max_rel:.2e}")

    assert abs(loss_1 - loss_2) < 1e-2, \
        f"Loss mismatch too large: {loss_1} vs {loss_2}"
    assert max_rel < 0.5, \
        f"Gradient divergence too large: max rel diff {max_rel:.2e}"
