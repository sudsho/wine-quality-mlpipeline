"""Data loading utilities for the wine quality dataset."""
import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read(path):
    df = pd.read_csv(path, sep=";")
    # The UCI csvs use spaces in column names ("fixed acidity" etc.). Normalise
    # to lowercase here too in case anyone ever ships a copy without the typo.
    df.columns = [c.strip() for c in df.columns]
    return df


def load_red():
    """Load the UCI red wine quality csv as a DataFrame."""
    return _read(os.path.join(DATA_DIR, "winequality-red.csv"))


def load_white():
    """Load the UCI white wine quality csv as a DataFrame."""
    return _read(os.path.join(DATA_DIR, "winequality-white.csv"))


def load_combined():
    """Load both red and white wines, tag with a wine_type column."""
    red = load_red()
    red["wine_type"] = "red"
    white = load_white()
    white["wine_type"] = "white"
    return pd.concat([red, white], axis=0, ignore_index=True)


def make_binary_target(df, threshold=7):
    """Turn the quality column into a 0/1 'good wine' label.

    Quality >= threshold is considered good. The UCI dataset is heavily
    skewed toward 5 and 6 so picking 7 gives us a ~15% positive class.
    """
    df = df.copy()
    df["good"] = (df["quality"] >= threshold).astype(int)
    return df
