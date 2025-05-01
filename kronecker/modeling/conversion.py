"""Utilities for converting models to use Kronecker factored layers."""

import torch
import torch.nn as nn
from copy import deepcopy

from ..layers import KroneckerLinear


def convert_linear_to_kronecker(module, factor_dim=None):
    """
    Convert a nn.Linear layer to a KroneckerLinear layer.
    
    Args:
        module (nn.Linear): Linear module to convert
        factor_dim (int, optional): Dimension to use for factorization
        
    Returns:
        KroneckerLinear: Kronecker-factored version of the input module
    """
    if isinstance(module, nn.Linear):
        return KroneckerLinear.from_linear(module, factor_dim)
    return module


def convert_to_kronecker_model(model, factor_dim=None, linear_only=True):
    """
    Convert a model to use Kronecker-factored layers.
    
    Args:
        model: The model to convert
        factor_dim (int, optional): Dimension to use for factorization
        linear_only (bool): If True, only replace linear layers. If False, may also replace other layer types
            that have Kronecker equivalents.
            
    Returns:
        model: Converted model with Kronecker-factored layers
    """
    # Create a deep copy to avoid modifying the original model
    model = deepcopy(model)
    
    # Function to recursively replace modules
    def replace_modules(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Replace the module if it's a Linear layer
            if isinstance(child, nn.Linear):
                setattr(module, child_name, convert_linear_to_kronecker(child, factor_dim))
                print(f"Converted {full_name} to KroneckerLinear")
            else:
                # Recursively process child modules
                replace_modules(child, full_name)
    
    # Start the recursive replacement
    replace_modules(model)
    
    return model 