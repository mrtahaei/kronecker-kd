"""GKDTrainer implementation for knowledge distillation using TRL library."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead


class GKDTrainer:
    """
    General Knowledge Distillation Trainer (GKDTrainer).
    
    This trainer implements knowledge distillation from a teacher model to a
    student model using the TRL (Transformer Reinforcement Learning) library.
    """
    
    def __init__(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        alpha=0.5,
        temperature=2.0,
    ):
        """
        Initialize a GKDTrainer.
        
        Args:
            teacher_model: The teacher model for knowledge distillation
            student_model: The student model (with Kronecker factorization)
            training_args: Training arguments (TrainingArguments)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator function
            tokenizer: Tokenizer for processing inputs
            alpha: Weight for the distillation loss (0-1)
            temperature: Temperature for softening the teacher logits
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Freeze the teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Initialize the underlying trainer
        self.trainer = Trainer(
            model=self.student_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_loss=self._compute_kd_loss,
        )
    
    def _compute_kd_loss(self, model, inputs):
        """
        Compute the knowledge distillation loss.
        
        Args:
            model: Student model
            inputs: Batch of inputs
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Move teacher to the same device as student
        self.teacher_model.to(model.device)
        
        # Set models to appropriate modes
        self.teacher_model.eval()
        model.train()
        
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher outputs (no gradient tracking needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Task-specific loss (e.g., language modeling)
        task_loss = student_outputs.loss
        
        # Knowledge distillation loss
        # Soften the teacher and student logits
        soft_teacher_logits = teacher_logits / self.temperature
        soft_student_logits = student_logits / self.temperature
        
        # Calculate the KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(soft_student_logits, dim=-1),
            F.softmax(soft_teacher_logits, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Combine the losses
        # alpha controls the weight of the distillation loss
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * kd_loss
        
        return combined_loss
    
    def train(self):
        """Train the student model using knowledge distillation."""
        return self.trainer.train()
    
    def evaluate(self):
        """Evaluate the student model."""
        return self.trainer.evaluate()
    
    def save_model(self, output_dir):
        """Save the trained student model."""
        self.trainer.save_model(output_dir)
        
    @classmethod
    def from_pretrained(cls, teacher_model_id, student_model, *args, **kwargs):
        """
        Initialize a GKDTrainer with a pretrained teacher model.
        
        Args:
            teacher_model_id: Hugging Face model ID for the teacher
            student_model: Student model with Kronecker factorization
            *args, **kwargs: Additional arguments for the trainer
            
        Returns:
            GKDTrainer: Initialized trainer
        """
        from transformers import AutoModelForCausalLM
        
        # Load the teacher model
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id)
        
        return cls(teacher_model, student_model, *args, **kwargs) 