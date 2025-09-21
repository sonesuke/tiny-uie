# Makefile for tiny_uie project

# Variables
PYTHON := python
UV := uv
CLAUD_HOOKS := .claude/hooks/python-hooks.py

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  sync        - Sync dependencies with uv"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Lint code with ruff"
	@echo "  typecheck   - Type check with ty"
	@echo "  gitleaks    - Check for secrets with gitleaks"
	@echo "  test        - Run tests with pytest"
	@echo "  coverage    - Run tests with coverage report"
	@echo "  clean       - Clean build artifacts"
	@echo "  check       - Run commit checks (gitleaks, format, lint, typecheck)"
	@echo "  hooks       - Run Claude Code hooks check"
	@echo "  all         - Run format, lint, typecheck, and test"

# Install dependencies
.PHONY: install
install:
	$(UV) sync

# Sync dependencies with uv
.PHONY: sync
sync:
	$(UV) sync

# Format code with ruff
.PHONY: format
format:
	$(UV) run ruff format src tests

# Lint code with ruff
.PHONY: lint
lint:
	$(UV) run ruff check --fix src tests

# Type check with ty
.PHONY: typecheck
typecheck:
	$(UV) run ty check src

# Run tests with pytest
.PHONY: test
test:
	$(UV) run pytest

# Check for secrets with gitleaks
.PHONY: gitleaks
gitleaks:
	gitleaks detect --source=.

# Run Claude Code hooks check
.PHONY: hooks
hooks:
	@echo "Running Claude Code hooks..."
	$(UV) run python $(CLAUD_HOOKS)

# Run tests with coverage report
.PHONY: coverage
coverage:
	$(UV) run pytest --cov=src

# Run commit checks
.PHONY: check
check: gitleaks format lint typecheck

# Run format, lint, typecheck, and test
.PHONY: all
all: gitleaks format lint typecheck test

# Run all checks including hooks
.PHONY: all-with-hooks
all-with-hooks: gitleaks format lint typecheck test hooks

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf dist/
	rm -rf build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
