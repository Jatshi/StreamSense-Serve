# Contributing

Create a focused branch, keep raw media and credentials out of Git, and include tests for behavior
changes. Before opening a pull request, run:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov=streamsense --cov-report=term-missing --cov-fail-under=80
```

Benchmark changes must include the exact command, model revision, hardware, input hash, raw JSON,
and a caveat separating fixture results from general claims. New datasets must document source,
license or consent, split policy, and personally identifiable information.

