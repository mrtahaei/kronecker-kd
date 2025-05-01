#!/usr/bin/env python
"""
Example script for training a Kronecker-factored GPT-2 model with knowledge distillation.
"""

import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset

from kronecker.modeling import convert_to_kronecker_model
from kronecker.utils import calculate_parameter_reduction
from trainers import GKDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Kronecker-factored GPT-2 model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library)",
    )
    parser.add_argument(
        "--dataset_config_name", 
        type=str, 
        default="wikitext-2-raw-v1", 
        help="The configuration name of the dataset"
    )
    parser.add_argument(
        "--factor_dim",
        type=int,
        default=None,
        help="Dimension to use for Kronecker factorization (default: auto)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for distillation loss (0-1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for softening logits",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    # Preprocess the dataset
    def tokenize_function(examples):
        # Tokenize the texts
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    
    # Load the teacher model
    print(f"Loading the teacher model: {args.model_name_or_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # Create a student model by converting to Kronecker factorization
    print(f"Converting to Kronecker model with factor_dim={args.factor_dim}")
    student_model = convert_to_kronecker_model(teacher_model, factor_dim=args.factor_dim)
    
    # Calculate and display parameter reduction statistics
    stats = calculate_parameter_reduction(teacher_model, student_model)
    print("\nParameter reduction statistics:")
    print(f"Original model: {stats['original_params']:,} parameters")
    print(f"Kronecker model: {stats['kronecker_params']:,} parameters")
    print(f"Absolute reduction: {stats['reduction_absolute']:,} parameters")
    print(f"Percentage reduction: {stats['reduction_percentage']:.2f}%\n")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=1000,
        save_total_limit=2,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # Initialize the GKD trainer
    trainer = GKDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        training_args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        alpha=args.alpha,
        temperature=args.temperature,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main() 