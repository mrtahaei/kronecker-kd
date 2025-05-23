#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark and Correctness Testing for Kronecker Product Linear Layer Implementations

This script compares three implementations:
1. Regular nn.Linear with explicit Kronecker product weights
2. KroneckerLinear with efficient sum implementation
3. TritonKroneckerLinear with GPU acceleration

Metrics measured:
- Correctness: Output equivalence between implementations
- Memory usage: Static parameter count and peak memory during forward pass
- Latency: Forward pass speed for different configurations

Test configurations:
- Matrix sizes: [512, 1024, 2048, 4096, 8192]
- Batch sizes: [1, 8, 32, 128, 512]
- Compression factors: [2, 4, 8, 16]
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
from contextlib import contextmanager
import gc

# Import Kronecker implementations
from kronecker.layers.kronecker_linear import KroneckerLinear
from kronecker.layers.kronecker_triton_kernel_2 import TritonKroneckerLinear

@contextmanager
def measure_peak_memory():
    """Context manager to measure peak GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            return peak_mem
    else:
        try:
            yield
        finally:
            return 0  # Return 0 if CUDA is not available

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def generate_factors(dim: int, factor: int) -> Tuple[int, int]:
    """Generate factorization of a dimension."""
    factor1 = int(dim / factor)
    factor2 = int(factor)
    return (factor1, factor2)

def setup_models(in_dim: int, out_dim: int, compression: int, device: torch.device) -> Dict:
    """Setup models for comparison."""
    # Calculate factors
    in_factors = generate_factors(in_dim, compression)
    out_factors = generate_factors(out_dim, compression)
    
    # Regular Linear
    linear = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    
    # KroneckerLinear
    kron_linear = KroneckerLinear(
        in_features=in_dim,
        out_features=out_dim,
        in_factors=in_factors,
        out_factors=out_factors,
        bias=True,
        device=device
    )
    
    # TritonKroneckerLinear
    try:
        triton_kron_linear = None
        if device.type == 'cuda':
            triton_kron_linear = TritonKroneckerLinear(
                in_features=in_dim,
                out_features=out_dim,
                in_factors=in_factors,
                out_factors=out_factors,
                bias=True,
                device=device
            )
    except Exception as e:
        print(f"Triton initialization failed: {e}")
        exit()
    
    # Make sure the Kronecker linear layers use the same weights
    if triton_kron_linear is not None:
        # Copy A and B factors from kron_linear to triton_kron_linear
        with torch.no_grad():
            triton_kron_linear.kn_factor_A.copy_(kron_linear.kn_factor_A.squeeze(0))
            triton_kron_linear.kn_factor_B.copy_(kron_linear.kn_factor_B.squeeze(0))
            triton_kron_linear.bias.copy_(kron_linear.bias)
    
    # For regular linear, set weights as Kronecker product
    with torch.no_grad():
        W = torch.kron(
            kron_linear.kn_factor_A.squeeze(0),
            kron_linear.kn_factor_B.squeeze(0)
        )
        linear.weight.copy_(W)
        linear.bias.copy_(kron_linear.bias)
    
    return {
        'linear': linear,
        'kron_linear': kron_linear,
        'triton_kron_linear': triton_kron_linear
    }

def measure_latency(model: torch.nn.Module, inputs: torch.Tensor, n_runs: int = 500, warmup: int = 100) -> float:
    """Measure average latency of model forward pass."""
    if not torch.cuda.is_available():
        exit()
        # CPU timing
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inputs)
        
            # Measure
            start_time = time.time()
            for _ in range(n_runs):
                _ = model(inputs)
            end_time = time.time()
        
        return (end_time - start_time) / n_runs
    else:
        # GPU timing with CUDA events
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inputs)
        
            # Measure
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            for _ in range(n_runs):
                _ = model(inputs)
                
            end_event.record()
            torch.cuda.synchronize()
        
        return start_event.elapsed_time(end_event) / (n_runs * 1000)  # Convert to seconds

def verify_correctness(models: Dict, inputs: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-3) -> Dict:
    """Verify that all models produce the same output within tolerance."""
    results = {}
    with torch.no_grad():
        linear_out = models['linear'](inputs)
        kron_out = models['kron_linear'](inputs)
        
        # Linear vs KroneckerLinear
        kron_diff = torch.max(torch.abs(linear_out - kron_out))
        results['linear_vs_kron'] = {
            'max_diff': kron_diff.item(),
            'is_close': torch.allclose(linear_out, kron_out, atol=atol, rtol=rtol)
        }
        
        # Linear vs TritonKroneckerLinear
        if models['triton_kron_linear'] is not None:
            triton_out = models['triton_kron_linear'](inputs)
            triton_diff = torch.max(torch.abs(linear_out - triton_out))
            results['linear_vs_triton'] = {
                'max_diff': triton_diff.item(),
                'is_close': torch.allclose(linear_out, triton_out, atol=atol, rtol=rtol)
            }
    
    return results

def run_benchmark(matrix_sizes: List[int], batch_sizes: List[int], compression_factors: List[int], 
                 device: torch.device) -> Dict:
    """Run the complete benchmark suite."""
    results = []
    
    for matrix_size in matrix_sizes:
        for batch_size in batch_sizes:
            for compression in compression_factors:
                print(f"Running benchmark: matrix_size={matrix_size}, batch_size={batch_size}, compression={compression}")
                
                # Setup
                in_dim = out_dim = matrix_size
                models = setup_models(in_dim, out_dim, compression, device)
                inputs = torch.randn(batch_size, in_dim, device=device)
                
                # Parameter counts
                param_counts = {
                    name: count_parameters(model) 
                    for name, model in models.items() 
                    if model is not None
                }
                
                # Memory usage during forward pass
                memory_usage = {}
                for name, model in models.items():
                    if model is not None:
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.empty_cache()
                            gc.collect()
                            with torch.no_grad():
                                _ = model(inputs)
                            memory_usage[name] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                        else:
                            memory_usage[name] = 0
                
                # Latency
                latency = {}
                for name, model in models.items():
                    if model is not None:
                        latency[name] = measure_latency(model, inputs)
                
                # Correctness
                correctness = verify_correctness(models, inputs)
                
                # Collect results
                result = {
                    'matrix_size': matrix_size,
                    'batch_size': batch_size,
                    'compression': compression,
                    'param_counts': param_counts,
                    'memory_usage': memory_usage,
                    'latency': latency,
                    'correctness': correctness
                }
                
                results.append(result)
                
                # Clean up GPU memory
                del models, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
    
    return results

def save_results(results: List[Dict], output_dir: str):
    """Save benchmark results to CSV and generate plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results into DataFrames
    param_data = []
    memory_data = []
    latency_data = []
    
    for result in results:
        matrix_size = result['matrix_size']
        batch_size = result['batch_size']
        compression = result['compression']
        
        # Parameters
        for model_name, param_count in result['param_counts'].items():
            param_data.append({
                'matrix_size': matrix_size,
                'batch_size': batch_size,
                'compression': compression,
                'model': model_name,
                'param_count': param_count
            })
        
        # Memory usage
        for model_name, mem_usage in result['memory_usage'].items():
            memory_data.append({
                'matrix_size': matrix_size,
                'batch_size': batch_size,
                'compression': compression,
                'model': model_name,
                'memory_usage_mb': mem_usage
            })
        
        # Latency
        for model_name, latency_time in result['latency'].items():
            latency_data.append({
                'matrix_size': matrix_size,
                'batch_size': batch_size,
                'compression': compression,
                'model': model_name,
                'latency_s': latency_time
            })
    
    # Convert to DataFrames
    param_df = pd.DataFrame(param_data)
    memory_df = pd.DataFrame(memory_data)
    latency_df = pd.DataFrame(latency_data)
    
    # Save raw data
    param_df.to_csv(os.path.join(output_dir, 'parameters.csv'), index=False)
    memory_df.to_csv(os.path.join(output_dir, 'memory.csv'), index=False)
    latency_df.to_csv(os.path.join(output_dir, 'latency.csv'), index=False)
    
    # Generate plots
    # Parameter counts by matrix size and compression
    plt.figure(figsize=(12, 8))
    sns.barplot(data=param_df, x='matrix_size', y='param_count', hue='model')
    plt.title('Parameter Count by Matrix Size')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'param_count_by_size.png'))
    
    # Memory usage by matrix size and batch size
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=memory_df, x='matrix_size', y='memory_usage_mb', hue='model', style='compression')
    plt.title('Peak Memory Usage During Forward Pass')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
    
    # Latency by matrix size and batch size
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=latency_df, x='matrix_size', y='latency_s', hue='model', style='compression')
    plt.title('Forward Pass Latency')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'latency.png'))
    
    # Speedup relative to nn.Linear
    latency_pivot = latency_df.pivot_table(
        index=['matrix_size', 'batch_size', 'compression'],
        columns='model',
        values='latency_s'
    ).reset_index()
    
    latency_pivot['kron_speedup'] = latency_pivot['linear'] / latency_pivot['kron_linear']
    if 'triton_kron_linear' in latency_pivot.columns:
        latency_pivot['triton_speedup'] = latency_pivot['linear'] / latency_pivot['triton_kron_linear']
    
    speedup_df = pd.melt(
        latency_pivot, 
        id_vars=['matrix_size', 'batch_size', 'compression'],
        value_vars=['kron_speedup', 'triton_speedup'] if 'triton_kron_linear' in latency_pivot.columns else ['kron_speedup'],
        var_name='implementation',
        value_name='speedup'
    )
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=speedup_df, x='matrix_size', y='speedup', hue='implementation', style='compression')
    plt.title('Speedup Relative to nn.Linear')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig(os.path.join(output_dir, 'speedup.png'))

def main():
    # Setup configurations
    matrix_sizes = [1024, 2048, 4096, 9192]
    batch_sizes = [256]
    compression_factors = [2, 4, 8]
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on {device}")
    
    # Run benchmarks
    results = run_benchmark(matrix_sizes, batch_sizes, compression_factors, device)
    
    # Save results
    save_results(results, f'benchmark_results_batchsize_{batch_sizes[0]}')
    
    print("Benchmark completed. Results saved to benchmark_results/")

def quick_test():
    """Run a simple test to verify implementation correctness."""
    print("Running quick test of Kronecker implementations...")
    
    # Setup a simple test case
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = out_dim = 1024
    batch_size = 16
    compression = 4
    
    # Create models
    models = setup_models(in_dim, out_dim, compression, device)
    inputs = torch.randn(batch_size, in_dim, device=device)
    
    # Verify correctness
    correctness = verify_correctness(models, inputs)
    
    print("\nCorrectness check results:")
    for impl, result in correctness.items():
        status = "✓ PASS" if result['is_close'] else "✗ FAIL"
        print(f"{impl}: {status} (max difference: {result['max_diff']:.8f})")
    
    # Quick performance check
    warmup = 5
    n_runs = 20
    
    print("\nLatency comparison:")
    for name, model in models.items():
        if model is not None:
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(inputs)
                
                # Measure time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(n_runs):
                        _ = model(inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                else:
                    start_time = time.time()
                    for _ in range(n_runs):
                        _ = model(inputs)
                    end_time = time.time()
            
            avg_time = (end_time - start_time) / n_runs
            params = count_parameters(model)
            
            print(f"{name}: {avg_time:.6f} seconds, {params} parameters")
    
    print("\nQuick test completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Kronecker Linear Implementations")
    parser.add_argument("--quick", action="store_true", help="Run a quick test instead of full benchmark")
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        main()