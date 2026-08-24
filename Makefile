.PHONY: install run debug clean lint lint-strict

# Strip shell overrides so the settings in pyproject.toml and
# src/__init__.py win: .venv, .uv-cache and .hf-cache stay in-project.
UV = env -u VIRTUAL_ENV -u UV_CACHE_DIR -u HF_HOME uv

# flake8 cannot read pyproject.toml, so its excludes are passed on the
# command line. They mirror [tool.mypy]: the in-project .venv, the caches
# and the vendored llm_sdk workspace member are not our code.
FLAKE8_EXCLUDE = --extend-exclude=.venv,.uv-cache,.hf-cache,llm_sdk,data

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

clean:
	$(UV) run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(pathlib.Path(p), ignore_errors=True) for p in ('.mypy_cache', '.pytest_cache', '.ruff_cache')]"

lint:
	$(UV) run flake8 $(FLAKE8_EXCLUDE) .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 $(FLAKE8_EXCLUDE) .
	$(UV) run mypy . --strict
