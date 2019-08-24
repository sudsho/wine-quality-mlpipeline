"""Tests for the prediction helpers."""
import numpy as np
import pytest

from src.pipeline import FEATURES, build_pipeline
from src.predict import predict_one, predict_batch, predict_proba_one


@pytest.fixture
def fitted_pipe():
    rng = np.random.RandomState(7)
    X = rng.rand(150, len(FEATURES))
    y = (X[:, -1] > 0.5).astype(int)
    pipe = build_pipeline(n_estimators=20, random_state=7)
    pipe.fit(X, y)
    return pipe


def _make_sample(value=0.5):
    return {f: value for f in FEATURES}


def test_predict_one_returns_int(fitted_pipe):
    out = predict_one(fitted_pipe, _make_sample())
    assert isinstance(out, int)
    assert out in (0, 1)


def test_predict_proba_in_unit_interval(fitted_pipe):
    p = predict_proba_one(fitted_pipe, _make_sample())
    assert 0.0 <= p <= 1.0


def test_predict_batch_length_matches(fitted_pipe):
    samples = [_make_sample(0.1), _make_sample(0.5), _make_sample(0.9)]
    out = predict_batch(fitted_pipe, samples)
    assert len(out) == 3
    assert all(x in (0, 1) for x in out)
