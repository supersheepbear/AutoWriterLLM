.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint-check lint-all type-check check-deps update-deps clean-docs

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts
	@echo "All build, test, coverage and Python artifacts removed successfully."

clean-build: ## remove build artifacts
	rm -fr build/ || true
	rm -fr dist/ || true
	rm -fr .eggs/ || true
	find . -name '*.egg-info' -exec rm -fr {} + || true
	find . -name '*.egg' -type f -exec rm -f {} + || true
	@echo "Build artifacts removed successfully."

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} + || true
	find . -name '*.pyo' -exec rm -f {} + || true
	find . -name '*~' -exec rm -f {} + || true
	find . -name '__pycache__' -exec rm -fr {} + || true
	@echo "Python file artifacts removed successfully."

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/ || true
	rm -f .coverage || true
	rm -fr htmlcov/ || true
	rm -fr .pytest_cache || true
	@echo "Test and coverage artifacts removed successfully."

lint-check: ## check style with ruff
	ruff check src/autowriterllm tests
	@echo "Linting check completed successfully."

lint-all: ## automatically fix style issues with ruff
	ruff check --fix src/autowriterllm tests
	@echo "Automatic linting completed successfully."

test: ## run tests quickly with the default Python
	pytest
	@echo "Tests completed successfully."

test-all: ## run tests on every Python version with tox
	tox
	@echo "All tests completed successfully on every Python version."

coverage: ## check code coverage quickly with the default Python
	coverage run --source src/autowriterllm -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
	@echo "Code coverage check completed successfully."

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/autowriterllm.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ src/autowriterllm || { echo "sphinx-apidoc failed"; exit 1; }
	$(MAKE) -C docs clean
	$(MAKE) -C docs html || { echo "Sphinx build failed"; exit 1; }
	$(BROWSER) docs/_build/html/index.html
	@echo "Documentation generated successfully."
clean-docs: ## Remove documentation build artifacts
	rm -rf docs/_build
	@echo "Documentation build artifacts removed successfully."
servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .
	@echo "Documentation server started successfully."

release: dist ## package and upload a release
	# Use twine's command-line interface directly to avoid metadata issues
	python -m twine upload --verbose dist/* || { echo "PyPI upload failed"; exit 1; }
	@echo "Release uploaded successfully."

dist: clean ## builds source and wheel package
	python -m build
	ls -l dist
	@echo "Distribution packages built successfully."

install: ## install the package to the active Python's site-packages
	pip install .
	@echo "Package installed successfully."

type-check: ## Run static type checks using mypy
	mypy src/autowriterllm
	@echo "Type checking completed successfully."

