"""Train the wine quality pipeline.

Usage:
    python -m src.train --config configs/default.yaml
"""
import argparse
import logging
import os

import yaml
from sklearn.model_selection import train_test_split

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
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    log.info("test accuracy: %.4f", score)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
