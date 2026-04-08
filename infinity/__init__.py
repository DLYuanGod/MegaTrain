"""MegaTrain: RAM-Centric Architecture for LLM Training.

Supports any HuggingFace decoder-only model:
- Dense models: Llama 2/3/4, Qwen 2/3, Mistral, Phi, Gemma, etc.
- Hybrid attention: Qwen 3.5 (linear + full attention)
- MoE models: Mixtral, DeepSeek-MoE, Qwen3-Next
"""

from .model.cpu_master import CPUMasterModel
from .config.training import CPUMasterConfig
from .data.datasets import ChatDataset, MetaMathDataset, collate_fn

__version__ = "0.2.0"

__all__ = [
    "CPUMasterModel",
    "CPUMasterConfig",
    "ChatDataset",
    "MetaMathDataset",
    "collate_fn",
]
