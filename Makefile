.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	uv run python -c "import shutil; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for name in ('__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache') for path in Path('.').rglob(name)]"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
