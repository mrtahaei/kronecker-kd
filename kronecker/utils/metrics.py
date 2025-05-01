"""Utility functions for calculating metrics for Kronecker models."""

import torch
import torch.nn as nn

from ..layers import KroneckerLinear


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_parameter_reduction(original_model, kronecker_model):
    """
    Calculate the parameter reduction achieved by using Kronecker factorization.
    
    Args:
        original_model: Original model
        kronecker_model: Model with Kronecker-factored layers
        
    Returns:
        dict: Dictionary with parameter counts and reduction statistics
    """
    original_params = count_parameters(original_model)
    kronecker_params = count_parameters(kronecker_model)
    
    reduction_absolute = original_params - kronecker_params
    reduction_percentage = (reduction_absolute / original_params) * 100
    
    return {
        'original_params': original_params,
        'kronecker_params': kronecker_params,
        'reduction_absolute': reduction_absolute,
        'reduction_percentage': reduction_percentage
    }


def get_layer_parameter_stats(model):
    """
    Get parameter statistics for each layer in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary mapping layer names to parameter statistics
    """
    stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, KroneckerLinear)):
            if isinstance(module, nn.Linear):
                params = module.weight.numel()
                if module.bias is not None:
                    params += module.bias.numel()
                layer_type = "Linear"
            else:  # KroneckerLinear
                params = module.A.numel() + module.B.numel()
                if module.bias is not None:
                    params += module.bias.numel()
                layer_type = "KroneckerLinear"
                
            stats[name] = {
                'type': layer_type,
                'params': params
            }
            
    return stats 