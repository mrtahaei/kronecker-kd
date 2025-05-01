# Kronecker Knowledge Distillation (KroneckerKD)

This repository contains code for training models with Kronecker-factored layers using knowledge distillation techniques with the TRL library.

## Overview

KroneckerKD allows you to convert standard neural network models (from HuggingFace) to versions with Kronecker-factored layers, and then train these compressed models using knowledge distillation with the original model as the teacher.

## Project Structure

- `kronecker/` - Core implementation of Kronecker factorization
  - `layers/` - Kronecker-factored layer implementations
  - `modeling/` - Model architectures with Kronecker layers
  - `utils/` - Utility functions for Kronecker factorization
- `trainers/` - Custom trainers for knowledge distillation (GKDTrainer)
- `examples/` - Example scripts for running KD with different models
- `configs/` - Configuration files for training
- `data/` - Scripts for data processing (or placeholders for datasets)
- `scripts/` - Utility scripts for training, evaluation, etc.
- `tests/` - Unit tests for the codebase

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd kronecker-kd

# Install the package
pip install -e .
```

## Usage

```python
from kronecker.modeling import convert_to_kronecker_model
from transformers import AutoModelForCausalLM
from trainers import GKDTrainer

# Load the base model (teacher)
teacher_model = AutoModelForCausalLM.from_pretrained("your-teacher-model")

# Convert to a Kronecker model (student)
student_model = convert_to_kronecker_model(teacher_model)

# Setup the trainer
trainer = GKDTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    # Other training arguments
)

# Train the model
trainer.train()
```

## License

[Choose appropriate license] 