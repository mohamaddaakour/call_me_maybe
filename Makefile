.PHONY: install run debug clean lint lint-strict

UV = env -u VIRTUAL_ENV -u UV_CACHE_DIR -u HF_HOME uv

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

clean:
	$(UV) run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(pathlib.Path(p), ignore_errors=True) for p in ('.mypy_cache', '.pytest_cache', '.ruff_cache')]"

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
