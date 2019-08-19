"""Flask app exposing the trained wine quality pipeline.

Endpoints:
    POST /predict  - JSON in, JSON out
    GET  /health   - liveness check
"""
import logging
import os

from flask import Flask, jsonify, request

from src.predict import load_model, predict_one, predict_proba_one
from src.pipeline import FEATURES


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/pipeline.joblib")
_model = None


def get_model():
    global _model
    if _model is None:
        log.info("loading model from %s", MODEL_PATH)
        _model = load_model(MODEL_PATH)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": "missing fields", "fields": missing}), 400
    pipe = get_model()
    label = predict_one(pipe, payload)
    proba = predict_proba_one(pipe, payload)
    return jsonify({"good": label, "probability": proba})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
