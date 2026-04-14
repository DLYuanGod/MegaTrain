#!/usr/bin/env python3
"""
[DEPRECATED] Standalone MegaTrain GRPO — use VERL integration instead.

For production RL training, use the VERL-based scripts:

    CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
    CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen3_5_27b_megatrain.sh

The VERL integration provides SGLang FP8 rollout, proper weight sync,
ref_in_actor pointer swap, and full GRPO pipeline orchestration.

This standalone script is kept for reference only. It uses simulated
rollouts (no real LLM generation) and does not support FP8 inference.

Usage (reference only):
    python examples/rl/train_grpo.py --config examples/rl/configs/qwen3_5_27b_grpo.yaml
"""

import gc
import logging
import os
import sys
import time
import argparse
from collections import defaultdict

import numpy as np
import psutil
import torch
import yaml

from infinity.model.cpu_master import CPUMasterModel
from infinity.config.training import CPUMasterConfig

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
except ImportError:
    CPU_ADAM_AVAILABLE = False

try:
    from verl.utils.torch_functional import logprobs_from_logits
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False


# ── Utility functions ───────────────────────────────────────────────────

def logprobs_from_logits_simple(logits, labels):
    """Fallback log-prob computation when VERL is not installed."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(logits, input_ids):
    """Compute per-token log probabilities from logits."""
    input_ids_gpu = input_ids.to(logits.device)
    if VERL_AVAILABLE:
        return logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])
    else:
        return logprobs_from_logits_simple(logits[:, :-1, :], input_ids_gpu[:, 1:])


def compute_grpo_advantage(token_level_rewards, response_mask, group_ids, norm_by_std=True, epsilon=1e-6):
    """Compute GRPO advantage: normalize rewards within each group."""
    scores = token_level_rewards.sum(dim=-1)  # [batch]

    id2scores = defaultdict(list)
    for i in range(scores.shape[0]):
        id2scores[group_ids[i]].append((i, scores[i]))

    advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for idx, items in id2scores.items():
            vals = torch.stack([v for _, v in items])
            mean = vals.mean()
            std = vals.std() if len(vals) > 1 else torch.tensor(1.0)
            for i, val in items:
                if norm_by_std:
                    advantages[i] = (val - mean) / (std + epsilon)
                else:
                    advantages[i] = val - mean

    return advantages.unsqueeze(-1) * response_mask


def grpo_loss(log_prob, old_log_prob, ref_log_prob, advantages, response_mask,
              clip_range=0.2, kl_coef=0.001):
    """Compute GRPO loss: clipped policy gradient + KL regularization."""
    # Only compute on valid response tokens
    lp_diff = (log_prob - old_log_prob) * response_mask
    lp_diff = torch.clamp(lp_diff, -5.0, 5.0)
    ratio = torch.exp(lp_diff)
    ratio = torch.where(response_mask.bool(), ratio, torch.ones_like(ratio))

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    num_valid = response_mask.sum().clamp(min=1)
    pg_loss_mean = (pg_loss * response_mask).sum() / num_valid

    # KL penalty to reference
    ref_diff = (ref_log_prob - log_prob) * response_mask
    ref_diff = torch.clamp(ref_diff, -5.0, 5.0)
    kl = torch.exp(ref_diff) - ref_diff - 1.0
    kl_loss = (kl * response_mask).sum() / num_valid

    total_loss = pg_loss_mean + kl_coef * kl_loss

    # Metrics
    with torch.no_grad():
        valid_mask = response_mask.bool()
        clip_frac = ((ratio[valid_mask] - 1.0).abs() > clip_range).float().mean() if valid_mask.any() else torch.tensor(0.0)
        approx_kl = (0.5 * lp_diff.pow(2) * response_mask).sum() / num_valid

    return total_loss, {
        "pg_loss": pg_loss_mean.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "clip_frac": clip_frac.item(),
        "approx_kl": approx_kl.item(),
    }


# ── Data preparation ────────────────────────────────────────────────────

def load_gsm8k_prompts(tokenizer, num_prompts, max_prompt_len):
    """Load GSM8K prompts and tokenize them."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        raise RuntimeError(
            "Cannot load GSM8K dataset. Install `datasets` package or "
            "provide a local dataset path in config."
        )

    instruction = 'Let\'s think step by step and output the final answer after "####".'
    prompts = []
    answers = []

    for i in range(min(num_prompts, len(ds))):
        question = ds[i]["question"] + " " + instruction
        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[:max_prompt_len]
        prompts.append(prompt_tokens)
        answers.append(ds[i]["answer"])

    return prompts, answers


def prepare_grpo_batch(tokenizer, prompts, answers, n_per_prompt, max_prompt_len, max_response_len):
    """Prepare a batch for GRPO: tokenized prompts + simulated responses."""
    batch_size = len(prompts) * n_per_prompt
    seq_len = max_prompt_len + max_response_len
    pad_id = tokenizer.pad_token_id or 0

    all_input_ids = []
    all_attention_mask = []
    all_response_mask = []

    for p_idx, (prompt_tokens, answer) in enumerate(zip(prompts, answers)):
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        if len(answer_tokens) > max_response_len:
            answer_tokens = answer_tokens[:max_response_len]

        actual_prompt_len = len(prompt_tokens)

        for r_idx in range(n_per_prompt):
            resp_tokens = list(answer_tokens)
            # Vary responses for different rollouts
            if r_idx > 0:
                n_vary = max(1, len(resp_tokens) // 10)
                for v in range(n_vary):
                    pos = (r_idx * 7 + v * 13) % len(resp_tokens)
                    resp_tokens[pos] = (resp_tokens[pos] + r_idx * 17) % tokenizer.vocab_size

            resp_len = len(resp_tokens)
            if resp_len < max_response_len:
                resp_tokens += [pad_id] * (max_response_len - resp_len)
            else:
                resp_tokens = resp_tokens[:max_response_len]
                resp_len = max_response_len

            # Left-pad prompt
            if actual_prompt_len < max_prompt_len:
                padded_prompt = [pad_id] * (max_prompt_len - actual_prompt_len) + prompt_tokens
            else:
                padded_prompt = prompt_tokens[:max_prompt_len]

            full_seq = (padded_prompt + resp_tokens)[:seq_len]
            attn = ([0] * (max_prompt_len - actual_prompt_len) +
                    [1] * (actual_prompt_len + resp_len) +
                    [0] * (max_response_len - resp_len))[:seq_len]
            resp_m = ([0] * max_prompt_len +
                      [1] * resp_len +
                      [0] * (max_response_len - resp_len))[:seq_len]

            all_input_ids.append(full_seq)
            all_attention_mask.append(attn)
            all_response_mask.append(resp_m)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    response_mask = torch.tensor(all_response_mask, dtype=torch.float32)
    group_ids = np.repeat(np.arange(len(prompts)), n_per_prompt)

    # Simulate rewards
    token_level_rewards = torch.zeros(batch_size, seq_len)
    for i in range(batch_size):
        reward = (1.0 + torch.randn(1).item() * 0.1) if (i % n_per_prompt == 0) else (-0.5 + torch.randn(1).item() * 0.5)
        valid_resp = int(response_mask[i].sum().item())
        if valid_resp > 0:
            token_level_rewards[i, max_prompt_len + valid_resp - 1] = reward

    return input_ids, attention_mask, response_mask, group_ids, token_level_rewards


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training with MegaTrain")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num-steps", type=int, default=None, help="Override training steps")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    grpo_cfg = cfg["grpo"]
    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]
    mem_cfg = cfg["memory"]
    data_cfg = cfg["dataset"]
    log_cfg = cfg.get("logging", {})

    num_steps = args.num_steps or train_cfg["num_steps"]
    num_prompts = grpo_cfg["num_prompts_per_batch"]
    n_per_prompt = grpo_cfg["n_rollouts_per_prompt"]
    batch_size = num_prompts * n_per_prompt
    max_prompt_len = data_cfg.get("max_prompt_len", 256)
    max_response_len = data_cfg.get("max_response_len", 768)
    seq_len = max_prompt_len + max_response_len

    torch.manual_seed(train_cfg.get("seed", 42))

    log.info("=" * 70)
    log.info("MEGATRAIN GRPO: SINGLE-GPU RL TRAINING")
    log.info("=" * 70)
    log.info(f"  Model:             {model_cfg['name']}")
    log.info(f"  Batch:             {batch_size} ({num_prompts} prompts x {n_per_prompt} rollouts)")
    log.info(f"  Seq length:        {seq_len} (prompt={max_prompt_len} + response={max_response_len})")
    log.info(f"  Steps:             {num_steps}")
    log.info(f"  LR:                {train_cfg['learning_rate']}")
    log.info(f"  KL coef:           {grpo_cfg['kl_loss_coef']}")
    log.info(f"  Clip range:        {grpo_cfg['clip_range']}")

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build MegaTrain model ──
    log.info("\nLoading model ...")
    t0 = time.time()

    dtype = torch.bfloat16 if model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16

    megatrain_config = CPUMasterConfig(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", 0),
        dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dataset_path="__grpo__",
        dataset_name="",
        max_seq_len=seq_len,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_steps=num_steps,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        checkpoint_interval=mem_cfg.get("checkpoint_interval", 4),
        num_grad_slabs=mem_cfg.get("num_grad_slabs", 12),
    )

    from transformers import AutoModelForCausalLM, AutoConfig
    hf_config = AutoConfig.from_pretrained(model_cfg["name"], trust_remote_code=True)
    is_vlm = hasattr(hf_config, 'vision_config')

    if is_vlm:
        try:
            from transformers import AutoModelForImageTextToText
            load_class = AutoModelForImageTextToText
        except ImportError:
            load_class = AutoModelForCausalLM
    else:
        load_class = AutoModelForCausalLM

    hf_model = load_class.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    model = CPUMasterModel(hf_model, megatrain_config)
    del hf_model

    # ── Optimizer ──
    if optim_cfg.get("type") == "deepspeed_adam" and CPU_ADAM_AVAILABLE:
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=train_cfg["learning_rate"],
            betas=(optim_cfg.get("beta1", 0.9), optim_cfg.get("beta2", 0.999)),
            eps=optim_cfg.get("eps", 1e-8),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            adamw_mode=True,
        )
        log.info("Using DeepSpeed CPUAdam optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=train_cfg["learning_rate"],
            betas=(optim_cfg.get("beta1", 0.9), optim_cfg.get("beta2", 0.999)),
            eps=optim_cfg.get("eps", 1e-8),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        log.info("Using PyTorch AdamW optimizer")

    torch.cuda.synchronize()
    t_init = time.time() - t0
    log.info(f"Model initialized in {t_init:.1f}s")

    # ── Load prompts ──
    log.info("Loading GSM8K prompts ...")
    total_prompts_needed = num_prompts * num_steps
    prompts, answers = load_gsm8k_prompts(tokenizer, total_prompts_needed, max_prompt_len)
    log.info(f"Loaded {len(prompts)} prompts")

    # ── Training loop ──
    log.info("=" * 70)
    log.info("Starting GRPO training ...")
    log.info("=" * 70)

    device = torch.device(f"cuda:{model_cfg.get('device', 0)}")
    process = psutil.Process()
    clip_range = grpo_cfg["clip_range"]
    kl_coef = grpo_cfg["kl_loss_coef"]
    max_grad_norm_val = train_cfg.get("max_grad_norm", 1.0)
    log_interval = log_cfg.get("log_interval", 1)

    all_metrics = []

    for step in range(num_steps):
        step_start = time.time()

        # Select prompts for this step
        start_idx = (step * num_prompts) % len(prompts)
        end_idx = start_idx + num_prompts
        if end_idx > len(prompts):
            start_idx = 0
            end_idx = num_prompts
        step_prompts = prompts[start_idx:end_idx]
        step_answers = answers[start_idx:end_idx]

        # Prepare batch
        input_ids, attention_mask, response_mask, group_ids, token_level_rewards = \
            prepare_grpo_batch(tokenizer, step_prompts, step_answers,
                               n_per_prompt, max_prompt_len, max_response_len)

        # 1. Compute old log_probs (actor forward-only)
        logits = model.forward_logits(input_ids, attention_mask)
        old_log_probs = compute_log_probs(logits, input_ids)
        old_log_probs = torch.cat([old_log_probs, torch.zeros(batch_size, 1, device=old_log_probs.device)], dim=1)
        del logits

        # 2. Compute ref log_probs (reference forward-only, same model for now)
        logits_ref = model.forward_logits(input_ids, attention_mask)
        ref_log_probs = compute_log_probs(logits_ref, input_ids)
        ref_log_probs = torch.cat([ref_log_probs, torch.zeros(batch_size, 1, device=ref_log_probs.device)], dim=1)
        del logits_ref

        # 3. Compute GRPO advantages
        advantages = compute_grpo_advantage(
            token_level_rewards, response_mask, group_ids,
            norm_by_std=grpo_cfg.get("advantage_norm", True),
        )

        # 4. Forward+backward with GRPO loss
        model.zero_grad()
        optimizer.zero_grad()

        old_lp_gpu = old_log_probs.detach().to(device)
        ref_lp_gpu = ref_log_probs.detach().to(device)
        adv_gpu = advantages.to(device)
        resp_mask_gpu = response_mask.to(device)

        del old_log_probs, ref_log_probs
        gc.collect()
        torch.cuda.empty_cache()

        def loss_fn(logits, input_ids_gpu):
            lp = compute_log_probs(logits, input_ids_gpu.to("cpu"))
            lp = torch.cat([lp, torch.zeros(batch_size, 1, device=lp.device)], dim=1)
            loss, metrics = grpo_loss(
                lp, old_lp_gpu, ref_lp_gpu, adv_gpu[:, 1:].contiguous(),
                resp_mask_gpu[:, 1:].contiguous(),
                clip_range=clip_range, kl_coef=kl_coef,
            )
            return loss, metrics

        loss_val, n_tokens, timing, meta = model.forward_and_backward_custom_loss(
            input_ids, attention_mask, loss_fn
        )

        # 5. Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_grad_norm_val)
        optimizer.step()
        model._sync_params_to_gpu()

        step_time = time.time() - step_start
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        cpu_mem = process.memory_info().rss / 1e9

        metrics = meta if meta else {}
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        metrics["step_time"] = step_time
        all_metrics.append(metrics)

        # Clean up
        del old_lp_gpu, ref_lp_gpu, adv_gpu, resp_mask_gpu
        gc.collect()
        torch.cuda.empty_cache()

        if (step + 1) % log_interval == 0:
            log.info(
                f"Step {step+1}/{num_steps} | "
                f"loss={metrics.get('total_loss', loss_val):.4f} | "
                f"pg={metrics.get('pg_loss', 0):.4f} | "
                f"kl={metrics.get('kl_loss', 0):.4f} | "
                f"grad={metrics.get('grad_norm', 0):.2f} | "
                f"clip={metrics.get('clip_frac', 0):.3f} | "
                f"time={step_time:.1f}s | "
                f"GPU={gpu_mem:.1f}GB | CPU={cpu_mem:.1f}GB"
            )

    # ── Summary ──
    log.info("")
    log.info("=" * 70)
    log.info("GRPO TRAINING COMPLETE")
    log.info("=" * 70)

    if all_metrics:
        avg_loss = np.mean([m.get("total_loss", 0) for m in all_metrics])
        avg_time = np.mean([m.get("step_time", 0) for m in all_metrics])
        first_loss = all_metrics[0].get("total_loss", 0)
        last_loss = all_metrics[-1].get("total_loss", 0)

        log.info(f"  Model:          {model_cfg['name']}")
        log.info(f"  Steps:          {num_steps}")
        log.info(f"  Batch:          {batch_size}")
        log.info(f"  Loss:           {first_loss:.4f} -> {last_loss:.4f}")
        log.info(f"  Avg loss:       {avg_loss:.4f}")
        log.info(f"  Avg step time:  {avg_time:.1f}s")
        log.info(f"  GPU peak:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model.cleanup()
    log.info("Done.")


if __name__ == "__main__":
    main()
