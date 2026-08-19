# wine-quality-mlpipeline

[![Build Status](https://travis-ci.org/sudsho/wine-quality-mlpipeline.svg?branch=master)](https://travis-ci.org/sudsho/wine-quality-mlpipeline)

Wine quality classifier built on the UCI Wine Quality dataset using a
scikit-learn `Pipeline`. Predicts whether a wine is "good" (quality >= 7).

## Quick start (runs offline)

No network, no downloads. The CSVs under `data/` are bundled, so the smoke
trains the pipeline and exercises the Flask predict path end to end:

```
python scripts/smoke.py     # or: make smoke
```

Real output:

```
[1/4] loading bundled data (offline)
      loaded rows=6497 features=11 positives=1303
[2/4] building pipeline + training
      3-fold CV accuracy: mean=0.7953 std=0.0025
      held-out test accuracy: 0.7977
[3/4] persisting model + wiring Flask test client
[4/4] POST /predict on a held sample
      response: good=0 probability=0.1667 (true label=0)

SMOKE OK: train + serve/predict path verified offline.
```

Run the unit tests the same way (also offline):

```
pytest -q
```

```
13 passed in 2.50s
```

Verified with scikit-learn 1.8, pandas 2.3, numpy 2.4, Flask 3.1 on
Python 3.11. The bundled CSVs carry the real UCI schema so the demo is
self-contained. They are a synthetic stand-in for the full UCI download, so the
accuracy here is baseline-level (the positive class is a ~20% minority); swap in
the genuine UCI CSVs for the headline numbers in the Results section below.

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

## Results

Trained on the combined red+white set, target = quality >= 7.

```
              precision    recall  f1-score   support

           0      0.910     0.964     0.937      1108
           1      0.706     0.498     0.584       192

    accuracy                          0.895      1300
   macro avg      0.808     0.731     0.760      1300
weighted avg      0.880     0.895     0.884      1300
```

Best params from `GridSearchCV` (5 folds, F1):
`rf__n_estimators=200, rf__max_depth=20, rf__min_samples_split=2`.

The recall on the positive class is the weak spot, expected given that good
wines are roughly 15% of the data. Could try class weights or SMOTE next.

## Serving

After `python -m src.train ...` the fitted pipeline lives at
`models/pipeline.joblib`. Bring up the Flask app with:

```
export FLASK_APP=app.py
flask run
```

Then POST a sample to `/predict`:

```
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' \
    -d '{"fixed acidity": 7.4, "volatile acidity": 0.7, ...}'
```

`GET /features` returns the exact list of expected keys.

## Deploy (Heroku)

```
heroku create
git push heroku master
heroku ps:scale web=1
```

`Procfile` runs gunicorn with two workers, `runtime.txt` pins
Python 3.7.4. The model artifact is *not* checked in - run training inside the
release phase or upload the joblib via a build hook.

## Tests

```
pytest -v
```

CI runs against Python 3.6 and 3.7 on Travis (see `.travis.yml`).

## License

MIT. See `LICENSE`.
