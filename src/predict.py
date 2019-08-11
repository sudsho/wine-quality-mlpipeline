"""Helpers for loading the trained pipeline and predicting on a single sample."""
import os
import joblib
import numpy as np

from src.pipeline import FEATURES


DEFAULT_MODEL_PATH = os.path.join("models", "pipeline.joblib")


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a previously persisted Pipeline from disk."""
    return joblib.load(path)


def predict_one(pipe, sample):
    """Predict good/bad for a single sample given as a dict of feature -> value."""
    row = np.array([[sample[f] for f in FEATURES]])
    label = int(pipe.predict(row)[0])
    return label
