PYTHON ?= python

.PHONY: install test lint format serve smoke v2-config v2-smoke benchmark

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

v2-config:
	$(PYTHON) scripts/validate_v2_config.py

v2-smoke:
	PORTFOLIO_V2_MODE=smoke PYTHON_BIN=$(PYTHON) bash scripts/autodl_v2_run.sh

benchmark:
	$(PYTHON) scripts/routing_benchmark.py --fixture benchmarks/data/router_fixture.jsonl --output runs/routing.json
