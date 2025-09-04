#!/usr/bin/env python3
"""
TensorFuse Benchmark Results Viewer
====================================

This script provides a comprehensive view of TensorFuse benchmark results
including performance metrics, comparisons, and analysis.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def load_bert_results(results_dir: str) -> Optional[Dict]:
    """Load BERT baseline benchmark results."""
    bert_file = os.path.join(results_dir, "bert_baseline_results.json")
    if not os.path.exists(bert_file):
        return None
    
    with open(bert_file, 'r') as f:
        return json.load(f)

def load_roofline_data(results_dir: str) -> Optional[List[Tuple[str, float, float]]]:
    """Load roofline analysis data."""
    roofline_file = os.path.join(results_dir, "roofline_data.txt")
    if not os.path.exists(roofline_file):
        return None
    
    kernels = []
    with open(roofline_file, 'r') as f:
        lines = f.readlines()
        
    # Parse actual kernel performance data
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        if 'roofline' in line:
            continue
        
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                oi = float(parts[0])  # Operational Intensity
                perf = float(parts[1])  # Performance (TFLOPS)
                kernel_name = ' '.join(parts[2:])
                kernels.append((kernel_name, oi, perf))
            except ValueError:
                continue
    
    return kernels

def display_bert_results(bert_data: Dict):
    """Display BERT baseline benchmark results."""
    print("🤖 BERT-base Baseline Benchmark Results")
    print("=" * 50)
    
    config = bert_data["bert_base_benchmark"]["config"]
    results = bert_data["bert_base_benchmark"]["results"]
    
    print(f"📊 Configuration:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Sequence length: {config['seq_len']}")
    print(f"   Hidden dimension: {config['hidden_dim']}")
    print(f"   FFN dimension: {config['ffn_dim']}")
    print(f"   Iterations: {config['benchmark_iterations']}")
    print()
    
    print("🏆 Performance Results:")
    print(f"{'Method':<20} {'Time (ms)':<12} {'TFLOPS':<10} {'Bandwidth (GB/s)':<15} {'Speedup':<10}")
    print("-" * 75)
    
    baseline_time = None
    for result in results:
        name = result["name"]
        avg_time = result["avg_time_ms"]
        tflops = result["tflops"]
        bandwidth = result["memory_bandwidth_gbps"]
        
        if baseline_time is None:
            baseline_time = avg_time
            speedup = "1.00x"
        else:
            speedup = f"{baseline_time/avg_time:.2f}x"
        
        print(f"{name:<20} {avg_time:<12.3f} {tflops:<10.2f} {bandwidth:<15.2f} {speedup:<10}")
    
    print()
    
    # Calculate and display speedup
    if len(results) >= 2:
        baseline = results[0]
        tensorfuse = results[1]
        speedup = baseline["avg_time_ms"] / tensorfuse["avg_time_ms"]
        
        print("🚀 Performance Analysis:")
        print(f"   TensorFuse speedup: {speedup:.2f}x")
        print(f"   Target speedup: 2.0x - 7.0x")
        
        if speedup >= 2.0:
            print("   ✅ Meeting target performance!")
        else:
            print("   ❌ Below target - needs optimization")
            print(f"   📈 Need {2.0/speedup:.2f}x more performance to reach minimum target")

def display_roofline_results(roofline_data: List[Tuple[str, float, float]]):
    """Display roofline analysis results."""
    print("\n🏔️  Roofline Analysis Results")
    print("=" * 50)
    
    print(f"{'Kernel':<30} {'Op. Intensity':<15} {'Performance':<15} {'Comparison':<15}")
    print("-" * 75)
    
    # Group kernels by size
    kernel_groups = {}
    for kernel_name, oi, perf in roofline_data:
        # Extract size from kernel name (e.g., "TensorFuse_128x512x256")
        size = kernel_name.split('_')[-1] if '_' in kernel_name else "unknown"
        if size not in kernel_groups:
            kernel_groups[size] = []
        kernel_groups[size].append((kernel_name, oi, perf))
    
    for size in sorted(kernel_groups.keys()):
        print(f"\n📏 Matrix Size: {size}")
        kernels = kernel_groups[size]
        
        tensorfuse_perf = None
        cublas_perf = None
        
        for kernel_name, oi, perf in kernels:
            if "TensorFuse" in kernel_name:
                tensorfuse_perf = perf
            elif "cuBLAS" in kernel_name:
                cublas_perf = perf
            
            print(f"   {kernel_name:<26} {oi:<15.2f} {perf:<15.2f} TFLOPS")
        
        # Calculate speedup for this size
        if tensorfuse_perf and cublas_perf:
            speedup = tensorfuse_perf / cublas_perf
            status = "✅" if speedup >= 1.0 else "❌"
            print(f"   {status} Speedup: {speedup:.2f}x")

def display_summary(bert_data: Optional[Dict], roofline_data: Optional[List[Tuple[str, float, float]]]):
    """Display overall performance summary."""
    print("\n📈 TensorFuse Performance Summary")
    print("=" * 50)
    
    # Overall assessment
    if bert_data:
        results = bert_data["bert_base_benchmark"]["results"]
        if len(results) >= 2:
            baseline = results[0]
            tensorfuse = results[1]
            overall_speedup = baseline["avg_time_ms"] / tensorfuse["avg_time_ms"]
            
            print(f"🎯 Overall BERT-base Speedup: {overall_speedup:.2f}x")
            
            if overall_speedup >= 2.0:
                print("✅ Performance Goal: ACHIEVED")
            else:
                print("❌ Performance Goal: NEEDS WORK")
                print(f"   Target: 2.0x - 7.0x speedup")
                print(f"   Current: {overall_speedup:.2f}x")
    
    # Roofline analysis summary
    if roofline_data:
        print("\n🏔️  Roofline Analysis Summary:")
        tensorfuse_count = sum(1 for name, _, _ in roofline_data if "TensorFuse" in name)
        cublas_count = sum(1 for name, _, _ in roofline_data if "cuBLAS" in name)
        
        print(f"   Kernels analyzed: {tensorfuse_count} TensorFuse, {cublas_count} cuBLAS")
        
        # Calculate average performance
        tensorfuse_perfs = [perf for name, _, perf in roofline_data if "TensorFuse" in name]
        cublas_perfs = [perf for name, _, perf in roofline_data if "cuBLAS" in name]
        
        if tensorfuse_perfs and cublas_perfs:
            avg_tf_perf = sum(tensorfuse_perfs) / len(tensorfuse_perfs)
            avg_cublas_perf = sum(cublas_perfs) / len(cublas_perfs)
            avg_speedup = avg_tf_perf / avg_cublas_perf
            
            print(f"   Average TensorFuse performance: {avg_tf_perf:.2f} TFLOPS")
            print(f"   Average cuBLAS performance: {avg_cublas_perf:.2f} TFLOPS")
            print(f"   Average speedup: {avg_speedup:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="View TensorFuse benchmark results")
    parser.add_argument("--results-dir", default="results/benchmarks",
                       help="Directory containing benchmark results")
    parser.add_argument("--bert-only", action="store_true",
                       help="Show only BERT baseline results")
    parser.add_argument("--roofline-only", action="store_true",
                       help="Show only roofline analysis results")
    parser.add_argument("--summary", action="store_true",
                       help="Show only summary")
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        print("💡 Run benchmarks first with: ./scripts/benchmark.sh")
        return 1
    
    # Load data
    bert_data = load_bert_results(args.results_dir)
    roofline_data = load_roofline_data(args.results_dir)
    
    if not bert_data and not roofline_data:
        print("❌ No benchmark results found!")
        print("💡 Run benchmarks first with: ./scripts/benchmark.sh")
        return 1
    
    print("🚀 TensorFuse Benchmark Results")
    print("=" * 50)
    print(f"📁 Results from: {args.results_dir}")
    
    # Display results based on arguments
    if args.summary:
        display_summary(bert_data, roofline_data)
    elif args.bert_only:
        if bert_data:
            display_bert_results(bert_data)
        else:
            print("❌ No BERT results found!")
    elif args.roofline_only:
        if roofline_data:
            display_roofline_results(roofline_data)
        else:
            print("❌ No roofline data found!")
    else:
        # Show everything
        if bert_data:
            display_bert_results(bert_data)
        if roofline_data:
            display_roofline_results(roofline_data)
        if bert_data or roofline_data:
            display_summary(bert_data, roofline_data)
    
    print("\n💡 Use --help to see all viewing options")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 