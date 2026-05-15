lint:
	ruff check .
	ruff format --check .
	ty check .
	

build:
	uv build