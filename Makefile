lint:
	ruff check .
	ty check .

build:
	uv build