"""Tests for the sklearn pipeline construction and fit."""
import numpy as np
import pytest

from src.pipeline import FEATURES, build_pipeline


def _toy_xy(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, len(FEATURES))
    # Make y mildly dependent on the last column ("alcohol") so the rf has
    # something to latch onto.
    y = (X[:, -1] > 0.5).astype(int)
    return X, y


def test_build_pipeline_has_two_steps():
    pipe = build_pipeline()
    names = [s[0] for s in pipe.steps]
    assert names == ["scaler", "rf"]


def test_pipeline_fits_and_predicts():
    # Pin the seed at every level - the test was flaky once on CI when the
    # rf happened to land below the 0.7 cutoff.
    pipe = build_pipeline(n_estimators=50, random_state=0)
    X, y = _toy_xy(n=300, seed=0)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    # Must do strictly better than the all-zeros majority baseline.
    assert (preds == y).mean() > 0.7
