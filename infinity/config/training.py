"""Training configuration for CPU Master model."""

from dataclasses import dataclass, field
import torch


@dataclass
class CPUMasterConfig:
    """Configuration for CPUMasterModel training.

    Supports any HuggingFace decoder-only model (Llama, Qwen, Mistral, Phi, etc.).

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        max_seq_len: Maximum sequence length
        batch_size: Batch size per training step
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_steps: Total number of training steps
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        max_grad_norm: Maximum gradient norm for clipping
        device: CUDA device index
        dtype: Data type for GPU computations (bfloat16 or float16)
        seed: Random seed for reproducibility
        log_interval: Steps between logging
        checkpoint_interval: Layers between checkpoints (for gradient checkpointing)
        dataset_path: Path to training dataset
        enable_timing: Enable CUDA timing (adds sync overhead)
        num_grad_slabs: Number of gradient slab buffers (>= 2 * checkpoint_interval recommended)
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
        trust_remote_code: Trust remote code when loading HF models
        system_prompt: Optional system prompt for chat dataset (empty = no system message)
        query_field: Field name for user query in dataset
        response_field: Field name for assistant response in dataset
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    device: int = 0
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    # Dataset
    dataset_path: str = ""
    dataset_name: str = ""
    dataset_dir: str = "data"
    max_seq_len: int = 1024
    system_prompt: str = ""
    query_field: str = "query"
    response_field: str = "response"
    train_on_prompt: bool = False

    # VLM
    freeze_vision_encoder: bool = True
    freeze_projector: bool = False

    # Training
    batch_size: int = 96
    gradient_accumulation_steps: int = 1
    num_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Memory
    checkpoint_interval: int = 4
    num_grad_slabs: int = 12

    # Logging
    log_interval: int = 1
    enable_timing: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_grad_slabs < 2 * self.checkpoint_interval:
            import warnings
            warnings.warn(
                f"num_grad_slabs ({self.num_grad_slabs}) < 2 * checkpoint_interval "
                f"({2 * self.checkpoint_interval}). This may cause gradient slab starvation."
            )

        if not self.dataset_path and not self.dataset_name:
            raise ValueError("Must specify either dataset_path or dataset_name")

        valid_attn = ("flash_attention_2", "sdpa", "eager")
        if self.attn_implementation not in valid_attn:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn}, "
                f"got '{self.attn_implementation}'"
            )
