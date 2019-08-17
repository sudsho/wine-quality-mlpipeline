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


def predict_batch(pipe, samples):
    """Same idea but for a list of dicts."""
    rows = np.array([[s[f] for f in FEATURES] for s in samples])
    return [int(x) for x in pipe.predict(rows)]


def predict_proba_one(pipe, sample):
    """Probability of the positive (good) class for a single sample."""
    row = np.array([[sample[f] for f in FEATURES]])
    proba = pipe.predict_proba(row)[0]
    # Return P(good=1) which is the second column under sklearn's convention.
    return float(proba[1])
