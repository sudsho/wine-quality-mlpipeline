"""Build the sklearn Pipeline that we use for training and serving."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def build_pipeline(n_estimators=100, max_depth=None, random_state=42):
    """Return a fresh Pipeline that scales features then fits a random forest."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
