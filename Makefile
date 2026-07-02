# devagi Makefile
# Works on Linux/macOS `make` and on Windows via `make` in Git Bash / MSYS2 / Chocolatey make.
# On plain PowerShell, use scripts in `scripts/local/*.ps1` instead.

.PHONY: help install install-cpu install-cuda install-dev smoke test lint format clean tree check-bounds

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: install-cpu install-dev  ## Install for local (CPU) development

install-cpu:  ## Install CPU torch + base deps
	$(PIP) install -r requirements/base.txt -r requirements/cpu.txt

install-cuda:  ## Install CUDA 12.1 torch + base deps + triton (Linux+CUDA only)
	$(PIP) install -r requirements/base.txt -r requirements/cuda121.txt

install-dev:  ## Install development tools (pytest, ruff, black, mypy)
	$(PIP) install -r requirements/dev.txt

smoke:  ## Run 5-min local smoke test
	$(PYTHON) -m src.train --stage 0 --preset local_smoke --smoke-only

test:  ## Run unit tests
	$(PYTHON) -m pytest -x tests/

check-bounds:  ## Static check for bounded-design-axiom violations
	$(PYTHON) scripts/ci/check_bounded.py

lint:  ## Ruff lint
	$(PYTHON) -m ruff check src tests

format:  ## Black format
	$(PYTHON) -m black src tests

clean:  ## Remove caches
	@echo "Cleaning caches..."
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache']]"

tree:  ## Print project tree (needs `tree` binary)
	@tree -I '.venv|__pycache__|logs|checkpoints|data|.git|*.egg-info' -L 3
