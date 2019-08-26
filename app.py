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


@app.route("/features", methods=["GET"])
def features():
    """Tell the caller which keys the /predict endpoint expects."""
    return jsonify({"features": FEATURES})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": "missing fields", "fields": missing}), 400
    # Reject non-numeric values up front, otherwise sklearn raises a less
    # helpful error inside the scaler.
    bad = [f for f in FEATURES if not isinstance(payload[f], (int, float))]
    if bad:
        return jsonify({"error": "non-numeric fields", "fields": bad}), 400
    try:
        pipe = get_model()
        label = predict_one(pipe, payload)
        proba = predict_proba_one(pipe, payload)
    except Exception as exc:
        log.exception("prediction failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"good": label, "probability": proba})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
