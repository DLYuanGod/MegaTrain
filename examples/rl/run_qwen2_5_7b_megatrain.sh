#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# GRPO Training: Qwen2.5-7B on a Single GPU via VERL + MegaTrain
# ──────────────────────────────────────────────────────────────────────────────
#
# Architecture:
#   SGLang (FP8)  ── rollout inference on GPU  (~7 GB weights + KV cache)
#   MegaTrain     ── training on CPU→GPU       (~4-5 GB transient GPU buffers)
#   Both coexist on one GPU; no weight reload between phases.
#
# Hardware: Single H100/A100 80GB.
#
# Usage:
#   # Default (GSM8K, auto-download model from HuggingFace)
#   CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
#
#   # Custom model path and data
#   MODEL_PATH=/path/to/Qwen2.5-7B \
#   TRAIN_FILE=/path/to/train.parquet \
#   TEST_FILE=/path/to/test.parquet \
#   CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
#
#   # Override any VERL config via CLI (Hydra syntax)
#   CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh \
#       data.train_batch_size=16 actor_rollout_ref.rollout.n=4
# ──────────────────────────────────────────────────────────────────────────────

set -x

# ── Defaults (override via environment) ──────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
TRAIN_FILE=${TRAIN_FILE:-"$(pwd)/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$(pwd)/data/gsm8k/test.parquet"}

PROJECT_NAME=${PROJECT_NAME:-"GRPO-Qwen2_5-7B-MegaTrain"}
EXP_NAME=${EXP_NAME:-"grpo-7b-1gpu"}
LOG_DIR=${LOG_DIR:-"logs"}

# ── Derived ──────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}"

python3 -m verl.trainer.main_ppo \
    model_engine=megatrain \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=megatrain \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.megatrain.checkpoint_interval=4 \
    actor_rollout_ref.actor.megatrain.num_grad_slabs=12 \
    actor_rollout_ref.actor.megatrain.max_seq_len=1536 \
    actor_rollout_ref.actor.megatrain.attn_implementation=flash_attention_2 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=megatrain \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    "$@" 2>&1 | tee "${LOG_DIR}/grpo-qwen2_5-7b-${TIMESTAMP}.log"
