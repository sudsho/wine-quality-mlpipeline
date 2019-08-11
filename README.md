# wine-quality-mlpipeline

Wine quality classifier built on the UCI Wine Quality dataset using a
scikit-learn `Pipeline`. Predicts whether a wine is "good" (quality >= 7).

## Dataset

UCI Wine Quality (Cortez et al., 2009). Two CSVs:

* `data/winequality-red.csv` - 1599 red wines
* `data/winequality-white.csv` - 4898 white wines

Eleven physicochemical features (acidity, sulphates, alcohol, etc.) plus a
quality score from 0-10. Combined dataset is heavily skewed: most wines sit at
quality 5 or 6.

## Approach

A simple sklearn Pipeline:

```
StandardScaler -> RandomForestClassifier
```

Hyperparameters are picked by `GridSearchCV` (5-fold, F1 scoring). The fitted
pipeline is persisted with `joblib`.

## Layout

```
src/
    load_data.py    # csv loaders + binary-target helper
    pipeline.py     # build_pipeline()
    train.py        # GridSearchCV + persist
    predict.py      # load + predict_one
configs/default.yaml
data/winequality-red.csv
data/winequality-white.csv
```

## Setup

```
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
```
