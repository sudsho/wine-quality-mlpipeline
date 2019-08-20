"""Train the wine quality pipeline.

Usage:
    python -m src.train --config configs/default.yaml
"""
import argparse
import json
import logging
import os

import joblib
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from src.load_data import load_combined, load_red, load_white, make_binary_target
from src.pipeline import FEATURES, build_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_dataset(cfg):
    use = cfg["data"]["use"]
    if use == "red":
        df = load_red()
    elif use == "white":
        df = load_white()
    else:
        df = load_combined()
    df = make_binary_target(df, threshold=cfg["data"]["threshold"])
    return df


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    df = load_dataset(cfg)
    log.info("loaded %d rows", len(df))

    X = df[FEATURES].values
    y = df["good"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    log.info("train=%d test=%d positives=%d", len(X_train), len(X_test), int(y.sum()))

    pipe = build_pipeline(random_state=cfg["model"]["random_state"])
    grid = GridSearchCV(
        pipe,
        param_grid=cfg["param_grid"],
        cv=cfg["cv"]["folds"],
        scoring=cfg["cv"]["scoring"],
        n_jobs=cfg["cv"]["n_jobs"],
    )
    grid.fit(X_train, y_train)

    log.info("best CV %s: %.4f", cfg["cv"]["scoring"], grid.best_score_)
    log.info("best params: %s", grid.best_params_)

    best = grid.best_estimator_
    test_score = best.score(X_test, y_test)
    log.info("test accuracy: %.4f", test_score)

    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    artifact_path = cfg["artifact"]["path"]
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(best, artifact_path)
    log.info("saved fitted pipeline to %s", artifact_path)

    # Dump a small training summary alongside the model so we can sanity check
    # later runs without re-loading the joblib.
    summary = {
        "best_score": float(grid.best_score_),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "test_accuracy": float(test_score),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    summary_path = os.path.join(os.path.dirname(artifact_path), "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
