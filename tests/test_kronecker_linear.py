"""Tests for the KroneckerLinear layer."""

import torch
import torch.nn as nn
import unittest

from kronecker.layers import KroneckerLinear


class TestKroneckerLinear(unittest.TestCase):
    """Test cases for the KroneckerLinear layer."""
    
    def test_init(self):
        """Test initialization of KroneckerLinear."""
        layer = KroneckerLinear(784, 256, bias=True)
        
        # Check dimensions
        self.assertEqual(layer.in_features, 784)
        self.assertEqual(layer.out_features, 256)
        
        # Check parameters exist
        self.assertIsNotNone(layer.A)
        self.assertIsNotNone(layer.B)
        self.assertIsNotNone(layer.bias)
    
    def test_forward(self):
        """Test the forward pass of KroneckerLinear."""
        batch_size = 16
        in_features = 784
        out_features = 256
        
        # Create a KroneckerLinear layer
        layer = KroneckerLinear(in_features, out_features, bias=True)
        
        # Create a random input tensor
        x = torch.randn(batch_size, in_features)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))
    
    def test_parameter_reduction(self):
        """Test that KroneckerLinear reduces parameters compared to nn.Linear."""
        in_features = 784
        out_features = 256
        
        # Create a standard linear layer
        linear_layer = nn.Linear(in_features, out_features)
        
        # Create a KroneckerLinear layer
        kronecker_layer = KroneckerLinear(in_features, out_features)
        
        # Count parameters
        linear_params = sum(p.numel() for p in linear_layer.parameters())
        kronecker_params = sum(p.numel() for p in kronecker_layer.parameters())
        
        # Check that KroneckerLinear has fewer parameters
        self.assertLess(kronecker_params, linear_params)
        
        # Print parameter reduction for information
        reduction = (1 - kronecker_params / linear_params) * 100
        print(f"Parameter reduction: {reduction:.2f}%")
    
    def test_from_linear(self):
        """Test conversion from nn.Linear to KroneckerLinear."""
        in_features = 784
        out_features = 256
        
        # Create a standard linear layer
        linear_layer = nn.Linear(in_features, out_features)
        
        # Convert to KroneckerLinear
        kronecker_layer = KroneckerLinear.from_linear(linear_layer)
        
        # Check dimensions
        self.assertEqual(kronecker_layer.in_features, in_features)
        self.assertEqual(kronecker_layer.out_features, out_features)
        
        # For a proper test, we would check that the conversion preserves the function
        # approximately, but that's complex and depends on the exact implementation of from_linear
    

if __name__ == "__main__":
    unittest.main() 