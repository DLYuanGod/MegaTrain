"""Tests for standalone HuggingFace checkpoint export from CPUMasterModel.

These tests deliberately avoid constructing a full ``CPUMasterModel``, whose
``__init__`` allocates GPU buffers and CUDA streams. ``save_checkpoint`` only
reads the CPU-side master weights (``embedding``, ``cpu_layers``, ``norm``,
``lm_head``, ``tied_lm_head``), so the CPU-master state is assembled the same
way ``__init__`` does and the real method is exercised against it. That keeps
the whole file runnable on any machine, with or without a GPU.
"""

import copy
import pathlib

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from infinity.model.cpu_master import CPUMasterModel, _discover_model_components


def _tiny_llama(tie_word_embeddings=False):
    """Build a small randomly-initialised Llama locally (no network access)."""
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=tie_word_embeddings,
    )
    return LlamaForCausalLM(config)


class _StubTokenizer:
    """Minimal stand-in that records that the tokenizer was persisted."""

    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir
        (pathlib.Path(output_dir) / "tokenizer_config.json").write_text("{}")


def _make_cpu_master(reference_model):
    """Assemble CPU-master state without running the GPU-allocating __init__.

    Mirrors CPUMasterModel.__init__ (cpu_master.py:377-396), which stores
    ``.cpu()`` copies of the discovered components, then perturbs the masters so
    they differ from the reference model's own weights. Without that
    perturbation a copy-master-into-shell bug would be invisible.
    """
    components = _discover_model_components(reference_model)

    model = object.__new__(CPUMasterModel)
    model.embedding = copy.deepcopy(components["embedding"]).cpu()
    model.norm = copy.deepcopy(components["norm"]).cpu() if components["norm"] else None
    model.lm_head = copy.deepcopy(components["lm_head"]).cpu()
    model.cpu_layers = [copy.deepcopy(layer).cpu() for layer in components["layers"]]

    model.tied_lm_head = components["lm_head"].weight is components["embedding"].weight
    if model.tied_lm_head:
        # __init__ (cpu_master.py:382-386) re-establishes tying on the CPU copies.
        model.lm_head.weight = model.embedding.weight

    # Make the masters distinguishable from the shell's weights.
    with torch.no_grad():
        for param in model.embedding.parameters():
            param.add_(1.0)
        for layer in model.cpu_layers:
            for param in layer.parameters():
                param.add_(2.0)
        if model.norm is not None:
            for param in model.norm.parameters():
                param.add_(3.0)
        if not model.tied_lm_head:
            for param in model.lm_head.parameters():
                param.add_(4.0)

    return model


def _stored_dtypes(checkpoint_dir):
    """Read tensor dtypes straight from the safetensors file.

    Avoids ``from_pretrained``'s dtype-promotion heuristics, so the assertion is
    about what was actually written to disk.
    """
    with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt") as f:
        return {key: f.get_tensor(key).dtype for key in f.keys()}


def test_cpu_master_exposes_save_checkpoint():
    assert hasattr(CPUMasterModel, "save_checkpoint"), (
        "CPUMasterModel has no save_checkpoint; fine-tuned weights cannot be "
        "persisted without going through VERL."
    )


def test_save_checkpoint_round_trips_through_transformers(tmp_path):
    """Master weights must land in a directory plain transformers can reload."""
    shell = _tiny_llama()
    master = _make_cpu_master(shell)
    tokenizer = _StubTokenizer()
    out = tmp_path / "ckpt"

    master.save_checkpoint(shell, tokenizer, str(out), save_bf16=False)

    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert tokenizer.saved_to == str(out)

    reloaded = AutoModelForCausalLM.from_pretrained(str(out))
    reloaded_components = _discover_model_components(reloaded)

    torch.testing.assert_close(
        reloaded_components["embedding"].weight,
        master.embedding.weight,
        msg="embedding master weights did not reach the checkpoint",
    )
    for idx, (layer_reloaded, layer_master) in enumerate(
        zip(reloaded_components["layers"], master.cpu_layers)
    ):
        for (name, p_reloaded), p_master in zip(
            layer_reloaded.named_parameters(), layer_master.parameters()
        ):
            torch.testing.assert_close(
                p_reloaded, p_master, msg=f"layer {idx} param {name} mismatch"
            )
    torch.testing.assert_close(
        reloaded_components["norm"].weight,
        master.norm.weight,
        msg="final norm master weights did not reach the checkpoint",
    )
    torch.testing.assert_close(
        reloaded_components["lm_head"].weight,
        master.lm_head.weight,
        msg="lm_head master weights did not reach the checkpoint",
    )


def test_save_checkpoint_handles_tied_lm_head(tmp_path):
    """With tied weights the shared tensor must be written once and stay shared.

    This is the subtle path: lm_head and embedding are the same tensor, so
    copying the lm_head master separately would double-apply an update.
    """
    shell = _tiny_llama(tie_word_embeddings=True)
    master = _make_cpu_master(shell)
    assert master.tied_lm_head, "fixture should produce a tied model"

    out = tmp_path / "ckpt_tied"
    master.save_checkpoint(shell, _StubTokenizer(), str(out), save_bf16=False)

    reloaded = AutoModelForCausalLM.from_pretrained(str(out))
    reloaded_components = _discover_model_components(reloaded)

    # The embedding must equal the master exactly: +1.0 applied once, not twice.
    torch.testing.assert_close(
        reloaded_components["embedding"].weight,
        master.embedding.weight,
        msg="tied embedding was not written exactly once",
    )
    assert (
        reloaded_components["lm_head"].weight
        is reloaded_components["embedding"].weight
    ), "tying was broken by the checkpoint round trip"


def test_save_checkpoint_save_bf16_casts_stored_weights(tmp_path):
    shell = _tiny_llama()
    master = _make_cpu_master(shell)
    out = tmp_path / "ckpt_bf16"

    master.save_checkpoint(shell, _StubTokenizer(), str(out), save_bf16=True)

    dtypes = set(_stored_dtypes(out).values())
    assert dtypes == {torch.bfloat16}, f"expected all-bfloat16 on disk, got {dtypes}"


def test_save_checkpoint_save_bf16_false_preserves_fp32(tmp_path):
    shell = _tiny_llama()
    master = _make_cpu_master(shell)
    out = tmp_path / "ckpt_fp32"

    master.save_checkpoint(shell, _StubTokenizer(), str(out), save_bf16=False)

    dtypes = set(_stored_dtypes(out).values())
    assert dtypes == {torch.float32}, f"expected all-float32 on disk, got {dtypes}"


def test_save_checkpoint_creates_missing_output_dir(tmp_path):
    shell = _tiny_llama()
    master = _make_cpu_master(shell)
    nested = tmp_path / "does" / "not" / "exist"

    master.save_checkpoint(shell, _StubTokenizer(), str(nested), save_bf16=False)

    assert (nested / "model.safetensors").exists()
