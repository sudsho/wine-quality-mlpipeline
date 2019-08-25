"""Tests for the dataset loaders."""
import pandas as pd

from src.load_data import (
    load_red,
    load_white,
    load_combined,
    make_binary_target,
)
from src.pipeline import FEATURES


def test_load_red_has_expected_columns():
    df = load_red()
    assert isinstance(df, pd.DataFrame)
    for f in FEATURES + ["quality"]:
        assert f in df.columns


def test_load_white_has_expected_columns():
    df = load_white()
    for f in FEATURES + ["quality"]:
        assert f in df.columns


def test_load_combined_has_wine_type():
    df = load_combined()
    assert "wine_type" in df.columns
    assert set(df["wine_type"].unique()) == {"red", "white"}


def test_make_binary_target_default_threshold():
    df = load_red()
    out = make_binary_target(df, threshold=7)
    assert "good" in out.columns
    assert set(out["good"].unique()).issubset({0, 1})
    # Threshold of 7 should yield a reasonable but minority positive class.
    pos = out["good"].mean()
    assert 0.0 < pos < 0.3
