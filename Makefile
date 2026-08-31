.PHONY: help setup smoke pilot full test lint clean paper appendix figures

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
RUN  ?= results/smoke

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create the venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt pytest
	$(PIP) install -q -e .
	@echo "ready. next: make smoke"

smoke: ## full pipeline offline -- no key, no network, no cost
	PYTHONPATH=src $(PY) -m sism.cli all --config configs/smoke.yaml

pilot: ## cheap live run on 15 items, to check cost before committing
	PYTHONPATH=src $(PY) -m sism.cli all --config configs/pilot.yaml

full: ## the paper's evaluation: 60 items, 5 models, 2 judges
	PYTHONPATH=src $(PY) -m sism.cli all --config configs/full.yaml

figures: ## re-render figures from an existing run
	PYTHONPATH=src $(PY) -m sism.cli figures --run $(RUN) --out paper/figures

appendix: ## regenerate the data-derived LaTeX appendices
	PYTHONPATH=src $(PY) -m sism.cli appendix --run $(RUN) --out paper/generated

paper: figures appendix ## rebuild every artifact the paper \input{}s
	@echo "figures -> paper/figures, appendices -> paper/generated"
	@echo "now compile paper/main.tex (latexmk -pdf main.tex)"

test: ## run the test suite
	PYTHONPATH=src $(PY) -m pytest tests -q

doctor: ## check credentials and provider reachability
	PYTHONPATH=src $(PY) -m sism.cli doctor

probes: ## probe-set stats and the manipulation check
	PYTHONPATH=src $(PY) -m sism.cli probes

clean: ## remove caches and generated results
	rm -rf .cache results/* paper/figures/* paper/generated/*
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
