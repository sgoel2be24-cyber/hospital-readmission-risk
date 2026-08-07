PY := .venv/bin/python

.PHONY: setup data tune train explain test api score all clean

## one-time environment setup
setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## download the UCI dataset into data/
data:
	mkdir -p data
	curl -sL -o data/dataset_diabetes.zip \
	  "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
	unzip -o -q data/dataset_diabetes.zip -d data/

## randomised hyperparameter search -> reports/best_params.json (~5 min)
tune:
	$(PY) -m src.tune

## train + compare all models, calibrate, write reports/ and models/ (~1 min)
train:
	$(PY) -m src.train

## ablation study: feature sets, native categoricals, ensembling (~2 min)
experiments:
	$(PY) -m src.experiments

## permutation importance + SHAP for the deployed model
explain:
	$(PY) -m src.explain

test:
	$(PY) -m pytest

## serve the scoring API on :8000 (docs at /docs)
api:
	.venv/bin/uvicorn src.api:app --reload --port 8000

## batch-score every encounter in the dataset
score:
	$(PY) -m src.predict

## full reproduction from a clean checkout
all: data train explain test

clean:
	rm -rf reports/figures/*.png reports/*.csv reports/metrics.json models/*.joblib
