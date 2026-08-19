"""Offline end to end smoke test for the wine quality pipeline.

Runs with no network and no external downloads. It:

  1. Loads the bundled UCI-schema CSVs from data/ (offline).
  2. Builds the sklearn Pipeline (StandardScaler -> RandomForestClassifier),
     reports a 3-fold cross-validated accuracy, fits on a train split, and
     prints the held-out test accuracy.
  3. Persists the fitted pipeline and exercises the serving path through the
     Flask test client: GET /health, GET /features, POST /predict.
  4. Asserts the predicted label is a valid class (0 or 1) and the probability
     is in [0, 1].

Run it with:  python scripts/smoke.py   (or: make smoke)
"""
import json
import os
import sys
import tempfile

# Make the repo root importable when run as `python scripts/smoke.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from src.load_data import load_combined, make_binary_target
from src.pipeline import FEATURES, build_pipeline


def main():
    print("[1/4] loading bundled data (offline)")
    df = make_binary_target(load_combined(), threshold=7)
    X = df[FEATURES].values
    y = df["good"].values
    print("      loaded rows=%d features=%d positives=%d"
          % (len(df), len(FEATURES), int(y.sum())))

    print("[2/4] building pipeline + training")
    pipe = build_pipeline(n_estimators=60, random_state=42)
    cv = cross_val_score(pipe, X, y, cv=3, scoring="accuracy", n_jobs=1)
    print("      3-fold CV accuracy: mean=%.4f std=%.4f" % (cv.mean(), cv.std()))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)
    print("      held-out test accuracy: %.4f" % test_acc)

    print("[3/4] persisting model + wiring Flask test client")
    tmp_dir = tempfile.mkdtemp(prefix="wine_smoke_")
    model_path = os.path.join(tmp_dir, "pipeline.joblib")
    joblib.dump(pipe, model_path)

    import app as app_module
    app_module.MODEL_PATH = model_path
    app_module._model = None
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok", "health failed"
    r = client.get("/features")
    assert r.get_json()["features"] == FEATURES, "features mismatch"

    print("[4/4] POST /predict on a held sample")
    sample = {f: float(v) for f, v in zip(FEATURES, X_te[0])}
    r = client.post(
        "/predict", data=json.dumps(sample), content_type="application/json"
    )
    assert r.status_code == 200, "predict returned %d" % r.status_code
    body = r.get_json()
    label = body["good"]
    proba = body["probability"]
    assert label in (0, 1), "invalid class: %r" % (label,)
    assert 0.0 <= proba <= 1.0, "probability out of range: %r" % (proba,)
    print("      response: good=%d probability=%.4f (true label=%d)"
          % (label, proba, int(y_te[0])))

    print("\nSMOKE OK: train + serve/predict path verified offline.")


if __name__ == "__main__":
    main()
