.RECIPEPREFIX = >
PYTHON ?= .venv/bin/python
.PHONY: setup setup-conda data features train eval app test lint

setup:
> python3 -m venv .venv
> $(PYTHON) -m pip install -e .

setup-conda:
> conda env create -f environment.yml || conda env update -f environment.yml

data:
> $(PYTHON) -m coolblocks.data.download --config configs/config.yaml

features:
> $(PYTHON) -m coolblocks.features.build --config configs/config.yaml

train:
> $(PYTHON) -m coolblocks.models.train --config configs/config.yaml

eval:
> $(PYTHON) -m coolblocks.evaluation.run_eval --config configs/config.yaml

app:
> $(PYTHON) -m streamlit run app/streamlit_app.py

test:
> $(PYTHON) -m pytest

lint:
> $(PYTHON) -m ruff check src tests app
