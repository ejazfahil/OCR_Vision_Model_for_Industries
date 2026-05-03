.PHONY: test lint clean

test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ --cov=src --cov-report=html

lint:
	python -m py_compile src/*.py && echo "All files OK"

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name "*.pyc" -delete
