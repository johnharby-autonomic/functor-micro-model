.PHONY: all clean test

all:
	@echo "Available targets:"
	@echo "  test"
	@echo "  clean"
	@echo "  package"

test:
	pytest -q -rs
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	find . -name ".DS_Store" -delete

package: clean test
	rm -rf release
	mkdir -p release
	cp -R . release/probability-functor-micro
