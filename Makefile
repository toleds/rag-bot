# Ensure Poetry is installed
.PHONY: ensure-poetry
ensure-poetry:
	@command -v poetry >/dev/null 2>&1 || { echo >&2 "Poetry is not installed. Please install it first."; exit 1; }

# Run checks: Ruff, Black (in check mode), and MyPy
.PHONY: check
check: ensure-poetry
	poetry run ruff check .
	poetry run black --check .

## Format files using black
.PHONY: format
format: ensure-poetry
	poetry run ruff check . --fix
	poetry run black .