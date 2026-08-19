.PHONY: smoke test train

# Fast, fully offline end to end check: train + serve/predict path.
smoke:
	python scripts/smoke.py

test:
	pytest -v

train:
	python -m src.train --config configs/default.yaml
