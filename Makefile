.PHONY: install experiments test lint clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

experiments:
	$(PYTHON) scripts/run_experiments.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .cache .pytest_cache .ruff_cache
