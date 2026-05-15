lint:
	ruff check .
	ruff format --check .
	ty check .

format:
	ruff format .

build:
	uv build