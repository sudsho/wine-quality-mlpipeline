# wine-quality-mlpipeline

[![Build Status](https://travis-ci.org/sudsho/wine-quality-mlpipeline.svg?branch=master)](https://travis-ci.org/sudsho/wine-quality-mlpipeline)

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
