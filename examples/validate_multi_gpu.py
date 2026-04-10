"""
Validate multi-GPU data parallelism by comparing single-GPU vs multi-GPU training.

Trains a small model (Qwen3.5-0.8B) on alpaca_en_demo for a few steps, then
asserts that per-step losses and final parameter checksums match within tolerance.

Usage:
    python examples/validate_multi_gpu.py --devices 0,1
    python examples/validate_multi_gpu.py --devices 0,1,2,3,4,5
"""

import argparse
import copy
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import CPUMasterConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
except ImportError:
    CPU_ADAM_AVAILABLE = False


def run_training(hf_model_name, tokenizer, hf_model_state, devices, batch_size, num_steps, seed):
    """Run training and return per-step losses + final param checksum."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    config = CPUMasterConfig(
        model_name=hf_model_name,
        devices=devices,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=1e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        checkpoint_interval=2,
        num_grad_slabs=6,
        dataset_name="alpaca_en_demo",
        dataset_dir="data",
        max_seq_len=256,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        seed=seed,
    )

    # Load fresh model from state dict
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    hf_model.load_state_dict(hf_model_state)

    model = CPUMasterModel(hf_model, config)
    del hf_model

    if CPU_ADAM_AVAILABLE:
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(), lr=config.learning_rate,
            betas=(config.beta1, config.beta2), eps=config.eps,
            weight_decay=config.weight_decay, adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW(
            model.get_parameters(), lr=config.learning_rate,
            betas=(config.beta1, config.beta2), eps=config.eps,
            weight_decay=config.weight_decay)

    dataset = ChatDataset(tokenizer, config.max_seq_len,
                          dataset_name="alpaca_en_demo", dataset_dir="data")
    # Use a generator with fixed seed for reproducible batching
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=0, shuffle=True, generator=g)
    data_iter = iter(dataloader)

    losses = []
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss_val, n_tokens, _ = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"])

        grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), config.max_grad_norm)
        optimizer.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        optimizer.zero_grad()

        losses.append(loss_val)
        logger.info(f"  [{len(devices)} GPU] Step {step+1}: loss={loss_val:.6f} grad_norm={grad_norm:.4f}")

    # Compute parameter checksum
    checksum = sum(p.data.float().sum().item() for p in model.get_parameters())

    model.cleanup()
    return losses, checksum


def main():
    parser = argparse.ArgumentParser(description="Validate multi-GPU numerical equivalence")
    parser.add_argument("--devices", type=str, required=True,
                        help="Comma-separated GPU device IDs for multi-GPU run (e.g., '0,1')")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Model to use for validation")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    multi_devices = [int(d) for d in args.devices.split(',')]
    world_size = len(multi_devices)
    batch_size = world_size * 2  # 2 samples per GPU
    single_device = [multi_devices[0]]

    logger.info("=" * 70)
    logger.info("MULTI-GPU VALIDATION")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {batch_size} (single GPU: {batch_size}, multi GPU: {batch_size // world_size} x {world_size})")
    logger.info(f"Steps: {args.num_steps}, Seed: {args.seed}")
    logger.info("=" * 70)

    # Load model + tokenizer once, save state dict for reproducible init
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    initial_state = copy.deepcopy(hf_model.state_dict())
    del hf_model

    # Run single-GPU
    logger.info("\n--- Single GPU run ---")
    single_losses, single_checksum = run_training(
        args.model, tokenizer, initial_state, single_device, batch_size, args.num_steps, args.seed)

    # Run multi-GPU
    logger.info(f"\n--- Multi GPU run ({world_size} GPUs) ---")
    multi_losses, multi_checksum = run_training(
        args.model, tokenizer, initial_state, multi_devices, batch_size, args.num_steps, args.seed)

    # Compare
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    loss_diffs = [abs(s - m) for s, m in zip(single_losses, multi_losses)]
    max_loss_diff = max(loss_diffs)
    checksum_diff = abs(single_checksum - multi_checksum)

    # Tolerance accounts for batch-size-dependent softmax numerics in bf16.
    # Different batch compositions change per-sample attention outputs by ~1e-3
    # due to different masked-value contributions to softmax normalizers.
    # This is inherent to transformer attention with padding, not a multi-GPU bug.
    loss_tol = 5e-2
    checksum_tol = 5.0

    for i, (sl, ml, diff) in enumerate(zip(single_losses, multi_losses, loss_diffs)):
        status = "OK" if diff < loss_tol else "MISMATCH"
        logger.info(f"  Step {i+1}: single={sl:.6f} multi={ml:.6f} diff={diff:.2e} [{status}]")

    logger.info(f"\nParam checksum: single={single_checksum:.4f} multi={multi_checksum:.4f} diff={checksum_diff:.2e}")

    if max_loss_diff < loss_tol and checksum_diff < checksum_tol:
        logger.info(f"\nPASSED: Max loss diff {max_loss_diff:.2e} < {loss_tol}, "
                    f"checksum diff {checksum_diff:.2e} < {checksum_tol}")
    else:
        logger.error(f"\nFAILED: Max loss diff {max_loss_diff:.2e} (tol {loss_tol}), "
                     f"checksum diff {checksum_diff:.2e} (tol {checksum_tol})")
        exit(1)


if __name__ == "__main__":
    main()
