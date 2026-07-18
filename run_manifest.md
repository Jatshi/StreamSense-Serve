# Run manifest

## Local foundation run

- Environment: Windows, Python 3.11.7
- Install: `python -m pip install -e ".[dev]"`
- Quality: `ruff check . && ruff format --check .`
- Test: `pytest --cov=streamsense --cov-report=term-missing`
- Result: 10 tests passed, 84% statement coverage
- GPU: not used
- Artifacts: terminal output only; remote/GPU run pending

