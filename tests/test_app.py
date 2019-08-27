"""Smoke tests for the Flask app."""
import json
import os
import tempfile

import joblib
import numpy as np
import pytest

import app as app_module
from src.pipeline import FEATURES, build_pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Train a tiny pipeline and dump it to a temp path that the app loads.
    rng = np.random.RandomState(0)
    X = rng.rand(100, len(FEATURES))
    y = (X[:, -1] > 0.5).astype(int)
    pipe = build_pipeline(n_estimators=10, random_state=0)
    pipe.fit(X, y)

    model_path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, str(model_path))

    monkeypatch.setattr(app_module, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(app_module, "_model", None)

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_features(client):
    r = client.get("/features")
    assert r.status_code == 200
    assert r.get_json()["features"] == FEATURES


def test_predict_happy_path(client):
    sample = {f: 0.5 for f in FEATURES}
    r = client.post("/predict", data=json.dumps(sample), content_type="application/json")
    assert r.status_code == 200
    body = r.get_json()
    assert body["good"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_missing_fields(client):
    sample = {f: 0.5 for f in FEATURES[:-2]}
    r = client.post("/predict", data=json.dumps(sample), content_type="application/json")
    assert r.status_code == 400
    body = r.get_json()
    assert body["error"] == "missing fields"
