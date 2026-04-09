"""Dataset implementations for MegaTrain.

Supports:
- Alpaca and ShareGPT data formats
- Multi-turn conversations (full history training)
- VLM image tokens in conversations
- Thinking mode (<think>...</think>) for reasoning models
- Train-on-prompt option
- Local JSON/JSONL, HuggingFace Hub, and legacy Arrow datasets
- LlamaFactory-compatible dataset_info.json registry
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FILEEXT2TYPE = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".txt": "text",
}

# Common image placeholder tokens across VLM models
IMAGE_PLACEHOLDER = "<image>"


def load_dataset_info(dataset_dir: str) -> dict:
    """Load dataset_info.json from a directory."""
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json not found in {dataset_dir}")
    with open(info_path, "r") as f:
        return json.load(f)


def load_dataset_by_name(dataset_name: str, dataset_dir: str = "data"):
    """Load a dataset by name from dataset_info.json."""
    from datasets import load_dataset, load_from_disk

    dataset_info = load_dataset_info(dataset_dir)
    if dataset_name not in dataset_info:
        available = list(dataset_info.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in dataset_info.json. "
            f"Available: {available}"
        )

    attr = dataset_info[dataset_name]
    split = attr.get("split", "train")
    num_samples = attr.get("num_samples", None)

    if "hf_hub_url" in attr:
        logger.info(f"Loading from HuggingFace Hub: {attr['hf_hub_url']}")
        kwargs = {}
        if "subset" in attr:
            kwargs["name"] = attr["subset"]
        ds = load_dataset(attr["hf_hub_url"], split=split, **kwargs)
    elif "file_name" in attr:
        file_path = os.path.join(dataset_dir, attr["file_name"])
        logger.info(f"Loading from local file: {file_path}")
        ext = Path(file_path).suffix.lower()
        if ext in FILEEXT2TYPE:
            ds = load_dataset(FILEEXT2TYPE[ext], data_files=file_path, split="train")
        elif os.path.isdir(file_path):
            ds = load_from_disk(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        raise ValueError(f"Dataset '{dataset_name}' must have 'hf_hub_url' or 'file_name'")

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    logger.info(f"Loaded {len(ds)} samples, columns: {ds.column_names}")
    return ds, attr


def convert_alpaca(sample: dict, columns: dict) -> list:
    """Convert alpaca-format sample to messages list.

    Returns list of {"role": ..., "content": ...} dicts.
    """
    prompt_col = columns.get("prompt", "instruction")
    query_col = columns.get("query", "input")
    response_col = columns.get("response", "output")
    system_col = columns.get("system", "system")
    images_col = columns.get("images", "images")

    instruction = sample.get(prompt_col, "")
    query = sample.get(query_col, "")
    response = sample.get(response_col, "")
    system = sample.get(system_col, "")
    images = sample.get(images_col, None)

    if query:
        user_content = f"{instruction}\n{query}"
    else:
        user_content = instruction

    # Add image placeholder for VLM
    if images:
        user_content = IMAGE_PLACEHOLDER + "\n" + user_content

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})

    return messages, images


def convert_sharegpt(sample: dict, columns: dict, tags: dict) -> list:
    """Convert sharegpt-format sample to messages list.

    Supports full multi-turn conversations.
    Returns list of {"role": ..., "content": ...} dicts.
    """
    messages_col = columns.get("messages", "conversations")
    system_col = columns.get("system", "system")
    images_col = columns.get("images", "images")

    role_tag = tags.get("role_tag", "from")
    content_tag = tags.get("content_tag", "value")
    user_tag = tags.get("user_tag", "human")
    assistant_tag = tags.get("assistant_tag", "gpt")
    system_tag = tags.get("system_tag", "system")

    raw_messages = sample.get(messages_col, [])
    system = sample.get(system_col, "")
    images = sample.get(images_col, None)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for msg in raw_messages:
        role = msg.get(role_tag, "")
        content = msg.get(content_tag, "")
        if role == system_tag:
            # System message: prepend or replace
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = content
            else:
                messages.insert(0, {"role": "system", "content": content})
        elif role == user_tag:
            messages.append({"role": "user", "content": content})
        elif role == assistant_tag:
            messages.append({"role": "assistant", "content": content})
        # Skip other roles (observation, function, etc. for now)

    # Add image placeholder to first user message for VLM
    if images:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = IMAGE_PLACEHOLDER + "\n" + msg["content"]
                break

    return messages, images


class ChatDataset(Dataset):
    """Universal SFT dataset with multi-turn, VLM, and thinking support.

    Features:
    - Multi-turn conversations (sharegpt format)
    - VLM image token handling
    - Thinking mode support (<think>...</think>)
    - Train-on-prompt option
    - Alpaca and sharegpt formats
    - LlamaFactory-compatible dataset_info.json

    Args:
        tokenizer: HuggingFace tokenizer (with chat_template support)
        max_seq_len: Maximum sequence length
        dataset_name: Dataset name in dataset_info.json
        dataset_dir: Directory with dataset_info.json
        dataset_path: Direct path to Arrow dataset (legacy)
        system_prompt: Optional system prompt override
        query_field: Field name for user query (legacy mode)
        response_field: Field name for assistant response (legacy mode)
        train_on_prompt: If True, don't mask prompt tokens (train on everything)
        processor: HF processor for VLM image preprocessing
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        dataset_name: str = None,
        dataset_dir: str = "data",
        dataset_path: str = None,
        system_prompt: str = None,
        query_field: str = "query",
        response_field: str = "response",
        train_on_prompt: bool = False,
        processor=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.train_on_prompt = train_on_prompt
        self.processor = processor

        if dataset_name:
            self._load_by_name(dataset_name, dataset_dir)
        elif dataset_path:
            self._load_by_path(dataset_path, query_field, response_field)
        else:
            raise ValueError("Must specify either dataset_name or dataset_path")

        logger.info(f"ChatDataset: {len(self.dataset)} samples, max_seq_len={max_seq_len}")
        if train_on_prompt:
            logger.info("  train_on_prompt=True (no prompt masking)")

    def _load_by_name(self, dataset_name: str, dataset_dir: str):
        """Load dataset via dataset_info.json registry."""
        self.dataset, self.attr = load_dataset_by_name(dataset_name, dataset_dir)
        self.formatting = self.attr.get("formatting", "alpaca")
        self.columns = self.attr.get("columns", {})
        self.tags = self.attr.get("tags", {})
        self.mode = "registry"

    def _load_by_path(self, dataset_path: str, query_field: str, response_field: str):
        """Load dataset from direct path (legacy Arrow format)."""
        from datasets import load_from_disk
        self.dataset = load_from_disk(dataset_path)
        self.query_field = query_field
        self.response_field = response_field
        self.mode = "legacy"

        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if self.query_field not in sample:
                available = list(sample.keys())
                raise ValueError(f"query_field '{self.query_field}' not found. Available: {available}")

    def _get_messages(self, idx: int):
        """Get messages list and optional images for a sample.

        Returns: (messages, images) where messages is a list of
                 {"role": ..., "content": ...} dicts
        """
        example = self.dataset[idx]
        images = None

        if self.mode == "legacy":
            query = example[self.query_field]
            response = example[self.response_field]
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": response})
            return messages, None

        if self.formatting == "sharegpt":
            messages, images = convert_sharegpt(example, self.columns, self.tags)
        else:
            messages, images = convert_alpaca(example, self.columns)

        if self.system_prompt:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = self.system_prompt
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages, images

    def _compute_labels(self, messages, input_ids, attention_mask):
        """Compute labels with proper masking for multi-turn conversations.

        For multi-turn, we mask all user/system turns and only supervise
        assistant turns. For train_on_prompt=True, we don't mask anything.
        """
        labels = input_ids.clone()

        if self.train_on_prompt:
            # Only mask padding
            labels[attention_mask == 0] = -100
            return labels

        # Compute prompt length: everything before the last assistant response
        # For multi-turn: mask all non-assistant content
        # Strategy: encode up to (but not including) the last assistant message,
        # that gives us the prompt length
        assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

        if not assistant_indices:
            # No assistant message, mask everything
            labels[:] = -100
            return labels

        # For single-turn or simple multi-turn: mask everything before last assistant
        # For full multi-turn training: mask all user/system turns
        prompt_messages = messages[:assistant_indices[-1]]
        if prompt_messages:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_encoded = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_length = int(prompt_encoded["attention_mask"].sum().item())
            labels[:prompt_length] = -100

        labels[attention_mask == 0] = -100
        return labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        messages, images = self._get_messages(idx)

        # Tokenize full conversation using tokenizer's native chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = self._compute_labels(messages, input_ids, attention_mask)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # VLM: process images if available
        if images and self.processor is not None:
            try:
                from PIL import Image
                # Load images if they're file paths
                if isinstance(images, list):
                    loaded = []
                    for img in images:
                        if isinstance(img, str):
                            loaded.append(Image.open(img).convert("RGB"))
                        else:
                            loaded.append(img)
                    images = loaded
                elif isinstance(images, str):
                    images = [Image.open(images).convert("RGB")]

                image_inputs = self.processor.image_processor(images, return_tensors="pt")
                for k, v in image_inputs.items():
                    result[k] = v.squeeze(0) if v.dim() > 3 else v
            except Exception as e:
                logger.warning(f"Failed to process images for sample {idx}: {e}")

        return result


# Backward-compatible alias
MetaMathDataset = ChatDataset


def collate_fn(batch):
    """Collate function for batching dataset samples.

    Handles variable keys (text-only vs VLM with pixel_values).
    """
    result = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }

    # Collate VLM image inputs if present
    if "pixel_values" in batch[0]:
        try:
            result["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        except RuntimeError:
            # Variable-size images: concat along batch dim
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)

    # Pass through any other vision kwargs (image_grid_thw, etc.)
    for key in batch[0]:
        if key not in result:
            try:
                result[key] = torch.stack([x[key] for x in batch])
            except (RuntimeError, TypeError):
                pass  # Skip non-stackable fields

    return result
