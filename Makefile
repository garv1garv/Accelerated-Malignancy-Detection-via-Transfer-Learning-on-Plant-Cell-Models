# Plant2Skin Transfer Learning Pipeline Makefile
# ⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️

.PHONY: help install prepare make_checkpoint train eval explain test all clean info

# Default target
help:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo ""
	@echo "Plant2Skin Transfer Learning Pipeline"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  prepare        - Generate synthetic dataset"
	@echo "  make_checkpoint - Create dummy plant checkpoint"
	@echo "  train          - Train the model"
	@echo "  eval           - Evaluate the model"
	@echo "  explain        - Generate Grad-CAM explanations"
	@echo "  test           - Run unit tests"
	@echo "  all            - Run complete pipeline"
	@echo "  clean          - Clean generated files"
	@echo "  info           - Show system information"
	@echo ""
	@echo "Examples:"
	@echo "  make all                    # Run complete pipeline"
	@echo "  make train epochs=5         # Train for 5 epochs"
	@echo "  make train freeze-backbone  # Train with frozen backbone"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Generate synthetic dataset
prepare:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Generating synthetic dataset..."
	python -m src.cli prepare
	@echo "✅ Dataset preparation completed!"

# Create dummy plant checkpoint
make_checkpoint:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Creating dummy plant checkpoint..."
	python -m src.cli make_checkpoint
	@echo "✅ Checkpoint creation completed!"

# Train the model
train:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Starting model training..."
	python -m src.cli train
	@echo "✅ Training completed!"

# Evaluate the model
eval:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Evaluating model..."
	python -m src.cli eval
	@echo "✅ Evaluation completed!"

# Generate explanations
explain:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Generating explanations..."
	python -m src.cli explain
	@echo "✅ Explanation generation completed!"

# Run tests
test:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Running tests..."
	pytest tests/ -v
	@echo "✅ Tests completed!"

# Run complete pipeline
all:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Running complete Plant2Skin transfer learning pipeline..."
	python -m src.cli all
	@echo "✅ Complete pipeline finished!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/synthetic/*
	rm -rf checkpoints/*
	rm -rf outputs/*
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	@echo "✅ Cleanup completed!"

# Show system information
info:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "System information:"
	python -m src.cli info

# Training with custom parameters
train-custom:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Training with custom parameters..."
	python -m src.cli train --epochs $(epochs) --batch-size $(batch_size) --learning-rate $(lr)
	@echo "✅ Custom training completed!"

# Quick demo (minimal training)
demo:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Running quick demo..."
	python -m src.cli all --epochs 2 --batch-size 8 --num-samples 40
	@echo "✅ Demo completed!"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	pip install pytest pytest-cov black flake8
	@echo "✅ Development setup completed!"

# Code formatting
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	@echo "✅ Code formatting completed!"

# Linting
lint:
	@echo "Running linter..."
	flake8 src/ tests/ scripts/
	@echo "✅ Linting completed!"

# Test with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated!"

# Documentation
docs:
	@echo "⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️"
	@echo "Documentation:"
	@echo "  - README.md: Main project documentation"
	@echo "  - ETHICS_AND_LIMITATIONS.md: Safety and ethical considerations"
	@echo "  - src/: Source code with inline documentation"
	@echo "  - tests/: Unit tests and examples"
	@echo ""
	@echo "⚠️  REMEMBER: This is a research prototype, not for clinical use!"

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import timm; print(f'timm: {timm.__version__}')"
	@python -c "import albumentations; print(f'albumentations: {albumentations.__version__}')"
	@echo "✅ System check completed!"

# Backup results
backup:
	@echo "Creating backup of results..."
	tar -czf plant2skin_results_$(shell date +%Y%m%d_%H%M%S).tar.gz outputs/
	@echo "✅ Backup created!"

# Show project structure
structure:
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|*.pyo|*.pyd|.git|.pytest_cache|*.egg-info' || find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" | grep -v __pycache__ | sort

# Validate setup
validate:
	@echo "Validating setup..."
	@python -c "from src.config import config; print('✅ Config loaded')"
	@python -c "from src.models import create_model; print('✅ Models module loaded')"
	@python -c "from src.data_utils import create_dataloaders; print('✅ Data utils loaded')"
	@python -c "from src.train import train_model; print('✅ Training module loaded')"
	@echo "✅ Setup validation completed!"
