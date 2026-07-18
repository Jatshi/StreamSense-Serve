.PHONY: install test lint format serve smoke

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=streamsense --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

serve:
	streamsense serve --host 127.0.0.1 --port 8000

smoke:
	python scripts/smoke_test.py

