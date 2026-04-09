"""
MegaTrain Training Script — Universal Single-GPU Large Model Training

Supports any HuggingFace decoder-only model: Llama 2/3/4, Qwen 2/3/3.5,
Mistral, DeepSeek, Phi, Gemma, and more.

Usage:
    # Train with YAML config
    python examples/train.py --config examples/configs/qwen_7b.yaml

    # Train with config + overrides
    python examples/train.py --config examples/configs/llama3_8b.yaml --batch-size 64 --num-steps 500
"""

import logging
import time
import os
import argparse
import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
    HAS_VLM_CLASS = True
except ImportError:
    HAS_VLM_CLASS = False

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import load_training_config, load_yaml_config, get_optimizer_type, get_num_workers, CPUMasterConfig

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Try to import DeepSpeed CPUAdam
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
    logger.info("DeepSpeed CPUAdam available (5-7x faster than PyTorch AdamW)!")
except ImportError:
    CPU_ADAM_AVAILABLE = False
    logger.info("DeepSpeed CPUAdam not available, using PyTorch AdamW")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train large language models with CPU-backed parameters")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset path from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override number of training steps from config")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_yaml_config(args.config)
        config = load_training_config(args.config)
        optimizer_type = get_optimizer_type(yaml_config)
        num_workers = get_num_workers(yaml_config)
    else:
        logger.info("Using default configuration")
        config = CPUMasterConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_seq_len=1024,
            batch_size=148,
            gradient_accumulation_steps=1,
            num_steps=100,
            learning_rate=1e-5,
            weight_decay=0.01,
            checkpoint_interval=4,
            dataset_path="dataset/Math/train",
            num_grad_slabs=12,
            device=0,
            dtype=torch.bfloat16,
            seed=42,
            log_interval=1,
        )
        optimizer_type = "deepspeed_adam"
        num_workers = 2

    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_steps:
        config.num_steps = args.num_steps

    logger.info("=" * 70)
    logger.info("MEGATRAIN: RAM-CENTRIC SINGLE-GPU TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Attention: {config.attn_implementation}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Training steps: {config.num_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")

    torch.manual_seed(config.seed)

    # Load tokenizer and detect VLM
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect VLM: check if model config has vision_config
    model_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    is_vlm = hasattr(model_config, 'vision_config') or (
        HAS_VLM_CLASS and type(model_config) in AutoModelForImageTextToText._model_mapping.keys()
    )

    processor = None
    if is_vlm:
        logger.info("VLM detected — loading with AutoModelForImageTextToText + AutoProcessor")
        processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        load_class = AutoModelForImageTextToText if HAS_VLM_CLASS else AutoModelForCausalLM
    else:
        load_class = AutoModelForCausalLM

    # Load model with specified attention implementation
    logger.info(f"Loading model with attn_implementation='{config.attn_implementation}'...")
    hf_model = load_class.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map="cpu",
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )

    # Verify attention implementation was applied
    attn_impl = getattr(hf_model.config, '_attn_implementation', 'unknown')
    logger.info(f"Model loaded. Attention implementation: {attn_impl}")
    if is_vlm:
        logger.info(f"VLM mode: vision encoder will be CPU-offloaded")

    # Create CPU Master model
    model = CPUMasterModel(hf_model, config)
    del hf_model

    # Setup optimizer
    if optimizer_type == "deepspeed_adam" and CPU_ADAM_AVAILABLE:
        logger.info("Using DeepSpeed CPUAdam optimizer (SIMD-accelerated)")
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            adamw_mode=True
        )
    else:
        logger.info("Using PyTorch AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    # Setup dataset (universal: uses tokenizer's native chat template)
    dataset_kwargs = dict(
        system_prompt=config.system_prompt if config.system_prompt else None,
        train_on_prompt=config.train_on_prompt,
        processor=processor,
    )
    if config.dataset_name:
        dataset = ChatDataset(
            tokenizer, config.max_seq_len,
            dataset_name=config.dataset_name,
            dataset_dir=config.dataset_dir,
            **dataset_kwargs,
        )
    else:
        dataset = ChatDataset(
            tokenizer, config.max_seq_len,
            dataset_path=config.dataset_path,
            query_field=config.query_field,
            response_field=config.response_field,
            **dataset_kwargs,
        )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    data_iter = iter(dataloader)

    # Training metrics
    total_loss = 0.0
    losses = []
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    throughputs = []
    gpu_mems = []
    cpu_mems = []
    process = psutil.Process()

    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    # Training loop
    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        start_time = time.perf_counter()

        # Forward and backward
        fwd_kwargs = {}
        if is_vlm and "pixel_values" in batch:
            fwd_kwargs["pixel_values"] = batch["pixel_values"]
            # Pass any additional vision kwargs (e.g., image_grid_thw for Qwen-VL)
            for k in batch:
                if k not in ("input_ids", "attention_mask", "labels", "pixel_values", "prompt_length"):
                    fwd_kwargs[k] = batch[k]

        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"], **fwd_kwargs
        )

        # Optimizer step
        if (step + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), config.max_grad_norm)
            optimizer.step()
            model._sync_params_to_gpu()
            model.zero_grad()
            optimizer.zero_grad()

        step_time = time.perf_counter() - start_time

        # Calculate metrics
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        cpu_mem = process.memory_info().rss / 1024**3

        num_params = sum(p.numel() for p in model.get_parameters())
        flops = 6 * num_params * n_tokens
        gflops = (flops / 1e9) / step_time

        fwd_time = timing['forward']
        bwd_time = timing['backward']

        step_times.append(step_time)
        throughputs.append(gflops)
        gpu_mems.append(gpu_mem)
        cpu_mems.append(cpu_mem)

        total_loss += loss_val
        losses.append(loss_val)

        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            tps = n_tokens / step_time
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3

            logger.info(f"Step {step+1}/{config.num_steps} | Loss {loss_val:.4f} | Avg {avg_loss:.4f}")
            logger.info(f"  Time: {step_time:.2f}s | Tokens/s {tps:.1f} | GFLOPS {gflops:.1f}")
            logger.info(f"  FWD: {fwd_time:.2f}s | BWD: {bwd_time:.2f}s")
            logger.info(f"  GPU: {gpu_mem:.2f}GB (alloc {mem_alloc:.2f}GB / reserved {mem_reserved:.2f}GB)")
            logger.info(f"  CPU: {cpu_mem:.2f}GB")

    # Training summary
    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Loss: {initial_loss:.4f} -> {final_loss:.4f} ({loss_reduction:.1f}% reduction)")
    logger.info(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"Peak CPU: {max(cpu_mems):.2f} GB")
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Avg Latency: {sum(step_times)/len(step_times):.3f}s per step")
    logger.info(f"  Avg Throughput: {sum(throughputs)/len(throughputs):.1f} GFLOPS")

    # Cleanup
    model.cleanup()


if __name__ == "__main__":
    main()
