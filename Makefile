# Makefile for Foresight SAR System
# Provides common development, testing, and deployment tasks

.PHONY: help install install-dev test test-unit test-integration test-acceptance
.PHONY: lint format security check quality coverage clean build docker
.PHONY: docs serve-docs deploy-docs sbom setup-dev setup-prod
.PHONY: start stop status logs backup restore

# Default target
help: ## Show this help message
	@echo "Foresight SAR System - Development Commands"
	@echo "==========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := foresight-sar
VERSION := $(shell $(PYTHON) -c "import sys; sys.path.append('.'); from main import __version__; print(__version__)" 2>/dev/null || echo "1.0.0")

# Installation
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	pre-commit install

setup-dev: install-dev ## Setup development environment
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	@mkdir -p data models evidence logs
	@echo "Development environment ready!"

setup-prod: install ## Setup production environment
	@echo "Setting up production environment..."
	@mkdir -p data models evidence logs
	@echo "Production environment ready!"

# Testing
test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	$(PYTEST) tests/unit/ -v --cov=. --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v --tb=short

test-acceptance: ## Run acceptance tests
	$(PYTEST) tests/acceptance/ -v --tb=short

test-performance: ## Run performance tests
	$(PYTEST) tests/ -m performance -v

test-gpu: ## Run GPU-specific tests
	$(PYTEST) tests/ -m gpu -v

test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ --tb=short -f

coverage: ## Generate coverage report
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint: ## Run linting
	flake8 .
	mypy .
	pylint **/*.py

format: ## Format code
	black .
	isort .
	autoflake --in-place --remove-all-unused-imports --recursive .

security: ## Run security checks
	bandit -r . -x tests/
	safety check
	pip-audit

check: lint security ## Run all code quality checks

quality: format lint security test-unit ## Run full quality pipeline

# Documentation
docs: ## Build documentation
	@echo "Building documentation..."
	@if [ -d "docs" ]; then \
		cd docs && make html; \
	else \
		echo "Documentation directory not found"; \
	fi

serve-docs: docs ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	@cd docs/_build/html && $(PYTHON) -m http.server 8000

deploy-docs: docs ## Deploy documentation
	@echo "Deploying documentation..."
	# Add your documentation deployment commands here

# Build and Package
build: clean ## Build the application
	$(PYTHON) -m build

sbom: ## Generate Software Bill of Materials
	$(PYTHON) tools/generate_sbom.py --output foresight-sar-sbom.json --summary
	@echo "SBOM generated: foresight-sar-sbom.json"

package: build sbom ## Create distribution package
	@echo "Creating distribution package..."
	@mkdir -p dist/$(PROJECT_NAME)-$(VERSION)
	@cp -r . dist/$(PROJECT_NAME)-$(VERSION)/
	@cd dist && tar -czf $(PROJECT_NAME)-$(VERSION).tar.gz $(PROJECT_NAME)-$(VERSION)/
	@echo "Package created: dist/$(PROJECT_NAME)-$(VERSION).tar.gz"

# Docker
docker-build: ## Build Docker image
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest

docker-run: ## Run Docker container
	$(DOCKER) run -d --name $(PROJECT_NAME) -p 8080:8080 $(PROJECT_NAME):latest

docker-stop: ## Stop Docker container
	$(DOCKER) stop $(PROJECT_NAME) || true
	$(DOCKER) rm $(PROJECT_NAME) || true

docker-compose-up: ## Start services with docker-compose
	$(DOCKER_COMPOSE) up -d

docker-compose-down: ## Stop services with docker-compose
	$(DOCKER_COMPOSE) down

docker-compose-logs: ## View docker-compose logs
	$(DOCKER_COMPOSE) logs -f

# Application Management
start: ## Start the application
	@echo "Starting Foresight SAR System..."
	$(PYTHON) main.py

start-simulate: ## Start with simulation mode
	@echo "Starting Foresight SAR System in simulation mode..."
	$(PYTHON) main.py --simulate

start-dev: ## Start in development mode
	@echo "Starting Foresight SAR System in development mode..."
	$(PYTHON) main.py --debug --simulate

stop: ## Stop the application
	@echo "Stopping Foresight SAR System..."
	@pkill -f "python main.py" || echo "No running instances found"

status: ## Check application status
	@echo "Checking Foresight SAR System status..."
	@pgrep -f "python main.py" > /dev/null && echo "Running" || echo "Stopped"

logs: ## View application logs
	@echo "Viewing logs..."
	@tail -f logs/foresight.log 2>/dev/null || echo "No log file found"

# Database and Data Management
backup: ## Backup data and models
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/foresight-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ models/ evidence/
	@echo "Backup created in backups/"

restore: ## Restore from backup (specify BACKUP_FILE=filename)
	@echo "Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE=filename"; exit 1; fi
	@tar -xzf backups/$(BACKUP_FILE)
	@echo "Backup restored"

# Deployment
deploy-windows: ## Deploy to Windows
	@echo "Deploying to Windows..."
	powershell -ExecutionPolicy Bypass -File deploy/windows/setup_windows.ps1

deploy-jetson: ## Deploy to Jetson
	@echo "Deploying to Jetson..."
	bash deploy/jetson/setup_jetson.sh

deploy-wizard: ## Run interactive deployment wizard
	$(PYTHON) deploy/setup_wizard.py

# Model Management
download-models: ## Download required models
	@echo "Downloading models..."
	@mkdir -p models
	$(PYTHON) -c "from vision.models import download_models; download_models()"

convert-models: ## Convert models to optimized formats
	@echo "Converting models..."
	bash tools/convert_to_tensorrt.sh

validate-models: ## Validate model files
	@echo "Validating models..."
	$(PYTHON) -c "from vision.models import validate_models; validate_models()"

# Monitoring and Maintenance
health-check: ## Run system health check
	@echo "Running health check..."
	$(PYTHON) -c "from utils.health import run_health_check; run_health_check()"

performance-test: ## Run performance benchmarks
	@echo "Running performance tests..."
	$(PYTEST) tests/ -m performance --benchmark-only

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete

clean-data: ## Clean temporary data files
	@echo "Cleaning temporary data..."
	rm -rf data/temp/ data/cache/
	rm -rf logs/*.log.* logs/temp/

clean-all: clean clean-data ## Clean everything
	@echo "Deep cleaning..."
	rm -rf .venv/ venv/
	rm -rf node_modules/

# Development Utilities
shell: ## Start Python shell with project context
	$(PYTHON) -i -c "import sys; sys.path.append('.'); print('Foresight SAR development shell')"

jupyter: ## Start Jupyter notebook
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats main.py --simulate
	@echo "Profile saved to profile.stats"

memory-profile: ## Profile memory usage
	mprof run main.py --simulate
	mprof plot

# Git and Version Control
git-setup: ## Setup git hooks and configuration
	pre-commit install
	git config --local core.autocrlf false
	git config --local core.eol lf

git-clean: ## Clean git repository
	git clean -fdx
	git reset --hard HEAD

# Environment Information
info: ## Show environment information
	@echo "Foresight SAR System Information"
	@echo "==============================="
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Project version: $(VERSION)"
	@echo "Working directory: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'unknown')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Virtual environment: $(shell echo $$VIRTUAL_ENV || echo 'none')"
	@echo "CUDA available: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"

# Quick development workflow
dev: setup-dev format lint test-unit ## Quick development setup and check

ci: format lint security test coverage sbom ## Full CI pipeline

# Release workflow
release: clean quality test package ## Prepare release
	@echo "Release $(VERSION) ready!"

# Default target when no arguments provided
.DEFAULT_GOAL := help