ENV=env
PIP=$(ENV)/bin/pip
RUFF=$(ENV)/bin/ruff
PYRIGHT=$(ENV)/bin/pyright

.PHONY: default run env install-dev check fmt lintfix lsp

default: run

check: env lsp fmt

env:
	$(info 🌍 ACTIVATING ENVIRONMENT...)
	@if [ ! -d "$(ENV)" ]; then python -m venv $(ENV); fi

install-dev: env
	$(info 📥 DOWNLOADING DEPENDENCIES...)
	$(PIP) install -r requirements_dev.txt

lsp: env
	$(info 🛠️ CHECKING STATIC TYPES...)
	$(PYRIGHT)

lintfix: env
	$(info 🔍 RUNNING LINT TOOLS...)
	$(RUFF) check --select I --fix

fmt: env
	$(info ✨ CHECKING CODE FORMATTING...)
	$(RUFF) format

run:
	$(info 🚀 RUNNING APP...)
	python python.py
