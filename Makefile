.RECIPEPREFIX = >
.PHONY: setup data features train eval app test lint

setup:
> conda env create -f environment.yml || conda env update -f environment.yml

data:
> python -m coolblocks.data.download --config configs/config.yaml

features:
> python -m coolblocks.features.build --config configs/config.yaml

train:
> python -m coolblocks.models.train --config configs/config.yaml

eval:
> python -m coolblocks.evaluation.run_eval --config configs/config.yaml

app:
> streamlit run app/streamlit_app.py

test:
> pytest

lint:
> ruff check src tests app
