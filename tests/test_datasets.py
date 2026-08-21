"""Tests for ChatDataset label masking, including RAG-style long system prompts.

A stub tokenizer is used rather than a pretrained one so the suite stays
offline and deterministic. The behaviour under test is token-budget
arithmetic, which does not depend on a particular vocabulary.
"""

import zlib

import pytest
import torch
from datasets import Dataset

from infinity.data.datasets import ChatDataset


class StubTokenizer:
    """Deterministic whitespace tokenizer with a minimal chat template."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|> {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def __call__(self, text, max_length=None, truncation=False, padding=None,
                 return_tensors=None, add_special_tokens=True):
        tokens = text.split()
        # crc32 keeps ids stable across runs (unlike hash(), which is salted).
        ids = [(zlib.crc32(t.encode()) % 1000) + 1 for t in tokens]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            padding_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * padding_len
            mask = mask + [0] * padding_len
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }


QUERY = "What is the capital of France?"
RESPONSE = "The capital of France is Paris."


@pytest.fixture
def make_dataset(tmp_path):
    """Build a real on-disk ChatDataset so __init__ and _load_by_path run."""

    def _make(system_prompt, max_seq_len, query=QUERY, response=RESPONSE):
        arrow_dir = tmp_path / f"ds_{abs(hash((system_prompt, max_seq_len)))}"
        Dataset.from_dict({"query": [query], "response": [response]}).save_to_disk(
            str(arrow_dir)
        )
        return ChatDataset(
            tokenizer=StubTokenizer(),
            max_seq_len=max_seq_len,
            dataset_path=str(arrow_dir),
            system_prompt=system_prompt,
        )

    return _make


def _supervised_token_count(sample):
    return int((sample["labels"] != -100).sum())


def test_oversized_system_prompt_still_supervises_response(make_dataset):
    """A RAG-style system message must not consume the whole budget.

    Retrieval context is packed into the system message and can exceed
    max_seq_len on its own. When that happens the conversation is truncated
    before the assistant turn, _compute_labels masks the full sequence, and the
    sample contributes no training signal at all.
    """
    huge_system = " ".join(f"ctx{i}" for i in range(400))
    dataset = make_dataset(huge_system, max_seq_len=128)

    supervised = _supervised_token_count(dataset[0])

    assert supervised > 0, (
        "every label is -100: the oversized system prompt pushed the assistant "
        "response out of the sequence, so this sample trains on nothing"
    )


def test_short_system_prompt_is_unaffected(make_dataset):
    """Samples that already fit must keep their existing supervision."""
    dataset = make_dataset("You are a helpful assistant.", max_seq_len=128)

    assert _supervised_token_count(dataset[0]) == len(RESPONSE.split())


def test_no_system_message_is_unaffected(make_dataset):
    """Datasets without a system turn must be left alone."""
    dataset = make_dataset(None, max_seq_len=128)

    assert _supervised_token_count(dataset[0]) == len(RESPONSE.split())


def test_truncation_shortens_only_the_system_turn(make_dataset):
    """The system content shrinks; user and assistant turns stay intact."""
    huge_system = " ".join(f"ctx{i}" for i in range(400))
    dataset = make_dataset(huge_system, max_seq_len=128)

    messages = [
        {"role": "system", "content": huge_system},
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": RESPONSE},
    ]
    result = dataset._truncate_system_for_response(messages)

    assert len(result) == 3
    assert len(result[0]["content"]) < len(huge_system), "system turn was not trimmed"
    assert result[1]["content"] == QUERY, "user turn must not be modified"
    assert result[2]["content"] == RESPONSE, "assistant turn must not be modified"


def test_truncation_does_not_mutate_caller_messages(make_dataset):
    """Trimming must not corrupt the caller's list, which callers may reuse."""
    huge_system = " ".join(f"ctx{i}" for i in range(400))
    dataset = make_dataset(huge_system, max_seq_len=128)

    messages = [
        {"role": "system", "content": huge_system},
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": RESPONSE},
    ]
    dataset._truncate_system_for_response(messages)

    assert messages[0]["content"] == huge_system, "input messages were mutated in place"


def test_truncation_is_a_noop_without_an_assistant_turn(make_dataset):
    """With nothing to supervise there is no budget to protect."""
    dataset = make_dataset("short", max_seq_len=128)

    messages = [
        {"role": "system", "content": " ".join(f"ctx{i}" for i in range(400))},
        {"role": "user", "content": QUERY},
    ]
    result = dataset._truncate_system_for_response(messages)

    assert result == messages
