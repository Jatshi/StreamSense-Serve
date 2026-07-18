PYTHON ?= python

.PHONY: install test lint format serve smoke benchmark

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=streamsense --cov-report=term-missing --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

serve:
	streamsense serve --host 127.0.0.1 --port 8000

smoke:
	$(PYTHON) scripts/smoke_test.py

benchmark:
	$(PYTHON) scripts/routing_benchmark.py --fixture benchmarks/data/router_fixture.jsonl --output runs/routing.json
