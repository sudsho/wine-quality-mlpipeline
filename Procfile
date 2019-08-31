release: python -m src.train --config configs/default.yaml
web: gunicorn app:app --workers=2 --timeout=120 --preload
