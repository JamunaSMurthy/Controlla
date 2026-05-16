"""Split and schema validation smoke tests."""

import pytest

from dataset_builder.schemas import SplitRatios


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_ratios_accept_valid_config() -> None:
    ratios = SplitRatios(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert ratios.prevent_identity_leakage is True