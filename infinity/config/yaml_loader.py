"""Configuration loading utilities for YAML config files."""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from infinity.config.training import CPUMasterConfig


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration values
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def yaml_to_training_config(yaml_config: Dict[str, Any]) -> CPUMasterConfig:
    """Convert YAML config to CPUMasterConfig.

    Args:
        yaml_config: Dictionary loaded from YAML file

    Returns:
        CPUMasterConfig instance
    """
    # Extract sections
    model_cfg = yaml_config.get('model', {})
    dataset_cfg = yaml_config.get('dataset', {})
    training_cfg = yaml_config.get('training', {})
    optimizer_cfg = yaml_config.get('optimizer', {})
    memory_cfg = yaml_config.get('memory', {})
    logging_cfg = yaml_config.get('logging', {})

    # Convert dtype string to torch.dtype
    dtype_str = model_cfg.get('dtype', 'bfloat16')
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Create CPUMasterConfig
    config = CPUMasterConfig(
        # Model
        model_name=model_cfg.get('name', 'Qwen/Qwen2.5-32B-Instruct'),
        device=model_cfg.get('device', 0),
        dtype=dtype,
        attn_implementation=model_cfg.get('attn_implementation', 'flash_attention_2'),
        trust_remote_code=model_cfg.get('trust_remote_code', True),

        # Dataset
        dataset_path=dataset_cfg.get('path', ''),
        dataset_name=dataset_cfg.get('name', ''),
        dataset_dir=dataset_cfg.get('dataset_dir', 'data'),
        max_seq_len=dataset_cfg.get('max_seq_len', 1024),
        system_prompt=dataset_cfg.get('system_prompt', ''),
        query_field=dataset_cfg.get('query_field', 'query'),
        response_field=dataset_cfg.get('response_field', 'response'),
        train_on_prompt=dataset_cfg.get('train_on_prompt', False),

        # VLM
        freeze_vision_encoder=yaml_config.get('vlm', {}).get('freeze_vision_encoder', True),
        freeze_projector=yaml_config.get('vlm', {}).get('freeze_projector', False),

        # Training
        batch_size=training_cfg.get('batch_size', 96),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        num_steps=training_cfg.get('num_steps', 100),
        learning_rate=training_cfg.get('learning_rate', 1e-5),
        weight_decay=training_cfg.get('weight_decay', 0.01),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        seed=training_cfg.get('seed', 42),

        # Optimizer
        beta1=optimizer_cfg.get('beta1', 0.9),
        beta2=optimizer_cfg.get('beta2', 0.999),
        eps=optimizer_cfg.get('eps', 1e-8),

        # Memory
        checkpoint_interval=memory_cfg.get('checkpoint_interval', 4),
        num_grad_slabs=memory_cfg.get('num_grad_slabs', 12),

        # Logging
        log_interval=logging_cfg.get('log_interval', 1),
        enable_timing=logging_cfg.get('enable_timing', True),
    )

    return config


def load_training_config(config_path: str) -> CPUMasterConfig:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        CPUMasterConfig instance
    """
    yaml_config = load_yaml_config(config_path)
    return yaml_to_training_config(yaml_config)


def get_optimizer_type(yaml_config: Dict[str, Any]) -> str:
    """Get optimizer type from YAML config.

    Returns:
        Optimizer type string ('deepspeed_adam' or 'adamw')
    """
    optimizer_cfg = yaml_config.get('optimizer', {})
    return optimizer_cfg.get('type', 'deepspeed_adam')


def get_num_workers(yaml_config: Dict[str, Any]) -> int:
    """Get number of DataLoader workers from YAML config.

    Returns:
        Number of workers
    """
    dataset_cfg = yaml_config.get('dataset', {})
    return dataset_cfg.get('num_workers', 2)
