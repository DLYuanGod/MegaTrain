"""Contracts for the deterministic train/validation split."""

import pytest


class FixtureDataset:
    """Minimal map-style dataset with a stable fingerprint."""

    _fingerprint = "fixture-v1"

    def __init__(self, size=40):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return index


def test_validation_split_is_deterministic_disjoint_ordered_and_capped():
    from infinity.data.datasets import split_dataset_for_validation

    dataset = FixtureDataset()
    train_a, validation_a, metadata_a = split_dataset_for_validation(
        dataset,
        split_fraction=0.5,
        seed=17,
        max_samples=7,
    )
    train_b, validation_b, metadata_b = split_dataset_for_validation(
        dataset,
        split_fraction=0.5,
        seed=17,
        max_samples=7,
    )
    _, validation_c, metadata_c = split_dataset_for_validation(
        dataset,
        split_fraction=0.5,
        seed=18,
        max_samples=7,
    )

    # max_samples caps the split: 50% of 40 would be 20, but 7 is the ceiling.
    assert len(train_a) == 33
    assert len(validation_a) == 7

    # Source order is preserved within each subset.
    assert train_a.indices == sorted(train_a.indices)
    assert validation_a.indices == sorted(validation_a.indices)

    # The two subsets partition the dataset exactly.
    assert set(train_a.indices).isdisjoint(validation_a.indices)
    assert set(train_a.indices) | set(validation_a.indices) == set(range(40))

    # Same seed reproduces the split and its fingerprint.
    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert metadata_a == metadata_b
    assert metadata_a["fingerprint"] == metadata_b["fingerprint"]

    # A different seed selects a different split.
    assert validation_a.indices != validation_c.indices
    assert metadata_a["fingerprint"] != metadata_c["fingerprint"]


def test_validation_split_requires_at_least_two_samples():
    from infinity.data.datasets import split_dataset_for_validation

    with pytest.raises(ValueError, match="at least two samples"):
        split_dataset_for_validation(
            ["only"],
            split_fraction=0.5,
            seed=42,
            max_samples=1,
        )


def test_validation_split_always_leaves_a_training_sample():
    """At the smallest workable size both subsets must be non-empty.

    split_fraction alone would claim the whole dataset here; validation is
    capped at dataset_size - 1 so training never ends up empty.
    """
    from infinity.data.datasets import split_dataset_for_validation

    train, validation, metadata = split_dataset_for_validation(
        FixtureDataset(size=2),
        split_fraction=1.0,
        seed=3,
        max_samples=100,
    )

    assert len(train) == 1
    assert len(validation) == 1
    assert metadata["train_samples"] == 1
    assert metadata["validation_samples"] == 1


def test_validation_split_fingerprint_tracks_split_parameters():
    """The fingerprint must distinguish runs that used different split settings."""
    from infinity.data.datasets import split_dataset_for_validation

    dataset = FixtureDataset()
    base = dict(split_fraction=0.25, seed=5, max_samples=100)

    _, _, metadata = split_dataset_for_validation(dataset, **base)
    _, _, other_fraction = split_dataset_for_validation(
        dataset, **{**base, "split_fraction": 0.5}
    )

    assert metadata["fingerprint"] != other_fraction["fingerprint"]
    assert metadata["validation_indices"] == tuple(
        sorted(metadata["validation_indices"])
    )
