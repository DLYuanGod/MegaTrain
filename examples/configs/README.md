# Configuration Guide

YAML configuration files for training models with MegaTrain.

## Training

```bash
# Step 1: Calculate optimal batch size for your hardware
python scripts/calc_resource.py

# Step 2: Train with a config file
python examples/train.py --config examples/configs/qwen_7b.yaml
python examples/train.py --config examples/configs/qwen3_8b.yaml
python examples/train.py --config examples/configs/qwen3_5_27b.yaml

# Override parameters from command line
python examples/train.py --config examples/configs/qwen_7b.yaml \
    --batch-size 64 --num-steps 500
```

> **IMPORTANT**: Always use `scripts/calc_resource.py` to determine the correct `batch_size` before training. Wrong batch size will cause OOM or waste GPU utilization.

## Available Configurations

| Config | Model | Architecture | Notes |
|:-------|:------|:-------------|:------|
| `qwen_7b.yaml` | Qwen 2.5 7B | Dense | Good for development and testing |
| `qwen3_8b.yaml` | Qwen 3 8B | Dense | Latest Qwen3 dense model |
| `qwen3_5_27b.yaml` | Qwen 3.5 27B | Hybrid (linear+full attn) | Requires `flash-linear-attention` |
| `qwen3_next_80b.yaml` | Qwen3-Next 80B-A3B | Hybrid + MoE | Large MoE model, needs 1TB+ RAM |
| `glm4_flash.yaml` | GLM-4-Flash 9B | Dense | Fast GLM model |
| `llama3_8b.yaml` | Llama 3.1 8B | Dense | Meta's Llama family |
| `gpt_oss_20b.yaml` | GPT-OSS 20B | Dense | OpenAI open-source model |

## Configuration Structure

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"     # HuggingFace model ID or local path
  dtype: "bfloat16"                      # bfloat16, float16, or float32
  device: 0                              # CUDA device index
  attn_implementation: "flash_attention_2"  # flash_attention_2, sdpa, or eager
  trust_remote_code: true

dataset:
  # Option 1: Dataset registry (recommended)
  name: "alpaca_en_demo"                 # Name from data/dataset_info.json
  dataset_dir: "data"                    # Directory with dataset_info.json
  # Option 2: Direct path (legacy Arrow format)
  # path: "/path/to/arrow/dataset"
  # query_field: "query"
  # response_field: "response"
  max_seq_len: 1024
  num_workers: 2

training:
  batch_size: 128                        # Use calc_resource.py to determine!
  gradient_accumulation_steps: 1
  num_steps: 500
  learning_rate: 1.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42

optimizer:
  type: "deepspeed_adam"                 # deepspeed_adam (fast) or adamw
  beta1: 0.9
  beta2: 0.999
  eps: 1.0e-8

memory:
  checkpoint_interval: 4                 # Layers between gradient checkpoints
  num_grad_slabs: 12                     # Must be >= 2 * checkpoint_interval

logging:
  log_interval: 1
  enable_timing: true
```

## Custom Configuration

1. Copy an existing config:
   ```bash
   cp examples/configs/qwen_7b.yaml examples/configs/my_model.yaml
   ```

2. Edit model name and dataset:
   ```yaml
   model:
     name: "your-org/your-model"
   dataset:
     name: "metamath"
   ```

3. Calculate batch size and train:
   ```bash
   python scripts/calc_resource.py
   python examples/train.py --config examples/configs/my_model.yaml
   ```

## Tips

- **OOM?** Reduce `batch_size`, increase `checkpoint_interval`, reduce `max_seq_len`
- **Slow?** Use `deepspeed_adam` optimizer, install `flash-attn`, increase `num_workers`
- **Qwen3.5 models?** Install `flash-linear-attention` + `causal-conv1d` for efficient linear attention
