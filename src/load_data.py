"""Data loading utilities for the wine quality dataset."""
import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_red():
    """Load the UCI red wine quality csv as a DataFrame."""
    path = os.path.join(DATA_DIR, "winequality-red.csv")
    return pd.read_csv(path, sep=";")


def load_white():
    """Load the UCI white wine quality csv as a DataFrame."""
    path = os.path.join(DATA_DIR, "winequality-white.csv")
    return pd.read_csv(path, sep=";")
