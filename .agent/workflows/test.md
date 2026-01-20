---
description: Run the project's test suite ensuring Python 3.12+ environment
---

# Run Tests

To run the tests, you must use the executable from the `.venv` directory to ensure Python 3.12+ is used.

## Run all tests

```bash
.venv/bin/pytest
```

## Run specific test file

```bash
.venv/bin/pytest tests/unit/test_statistics.py
```

## Run with verbose output

```bash
.venv/bin/pytest -v
```

**IMPORTANT**: Do not use `pytest` directly from the path, as it may point to the system Python (often 3.9 on Mac), which will cause syntax errors with the project's modern Python 3.12+ syntax.
