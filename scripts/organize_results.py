#!/usr/bin/env python3
"""
TensorFuse Results Organizer
============================

This script organizes benchmark results with timestamping and provides
historical comparison capabilities.
"""

import json
import os
import shutil
import argparse
from datetime import datetime
from typing import Dict, List, Optional

def get_timestamp() -> str:
    """Get current timestamp in a filename-friendly format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def copy_results_to_main(build_dir: str = "build/benchmarks/results", 
                        results_dir: str = "results/benchmarks") -> bool:
    """Copy results from build directory to main results directory."""
    if not os.path.exists(build_dir):
        print(f"❌ Build results directory not found: {build_dir}")
        return False
    
    # Create main results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy all files from build results
    copied_files = []
    for filename in os.listdir(build_dir):
        src = os.path.join(build_dir, filename)
        dst = os.path.join(results_dir, filename)
        
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied_files.append(filename)
    
    if copied_files:
        print(f"✅ Copied {len(copied_files)} result files to {results_dir}")
        for file in copied_files:
            print(f"   📄 {file}")
    
    return len(copied_files) > 0

def archive_results(results_dir: str = "results/benchmarks",
                   archive_dir: str = "results/archive") -> bool:
    """Archive current results with timestamp."""
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create timestamped archive directory
    timestamp = get_timestamp()
    archive_path = os.path.join(archive_dir, f"results_{timestamp}")
    
    try:
        shutil.copytree(results_dir, archive_path)
        print(f"✅ Results archived to: {archive_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to archive results: {e}")
        return False

def load_archived_results(archive_dir: str = "results/archive") -> Dict[str, Dict]:
    """Load all archived results for comparison."""
    if not os.path.exists(archive_dir):
        return {}
    
    archived_results = {}
    
    for archive_name in os.listdir(archive_dir):
        archive_path = os.path.join(archive_dir, archive_name)
        if not os.path.isdir(archive_path):
            continue
        
        # Try to load BERT results from this archive
        bert_file = os.path.join(archive_path, "bert_baseline_results.json")
        if os.path.exists(bert_file):
            try:
                with open(bert_file, 'r') as f:
                    bert_data = json.load(f)
                    archived_results[archive_name] = bert_data
            except Exception as e:
                print(f"Warning: Could not load {bert_file}: {e}")
    
    return archived_results

def compare_results(archived_results: Dict[str, Dict], 
                   current_results_dir: str = "results/benchmarks") -> None:
    """Compare current results with archived results."""
    print("\n📊 Historical Performance Comparison")
    print("=" * 60)
    
    # Load current results
    current_bert_file = os.path.join(current_results_dir, "bert_baseline_results.json")
    if not os.path.exists(current_bert_file):
        print("❌ No current BERT results found")
        return
    
    with open(current_bert_file, 'r') as f:
        current_bert = json.load(f)
    
    # Extract current performance
    current_results = current_bert["bert_base_benchmark"]["results"]
    current_speedup = None
    
    if len(current_results) >= 2:
        baseline = current_results[0]
        tensorfuse = current_results[1]
        current_speedup = baseline["avg_time_ms"] / tensorfuse["avg_time_ms"]
    
    if current_speedup is None:
        print("❌ Cannot calculate current speedup")
        return
    
    print(f"🎯 Current speedup: {current_speedup:.2f}x")
    print()
    
    # Compare with archived results
    if not archived_results:
        print("📝 No archived results for comparison")
        return
    
    print("📈 Historical Performance:")
    print(f"{'Archive':<20} {'Speedup':<10} {'Change':<10} {'Status':<15}")
    print("-" * 60)
    
    sorted_archives = sorted(archived_results.keys())
    for archive_name in sorted_archives:
        archive_data = archived_results[archive_name]
        
        # Extract archived performance
        archive_results = archive_data["bert_base_benchmark"]["results"]
        if len(archive_results) >= 2:
            baseline = archive_results[0]
            tensorfuse = archive_results[1]
            archive_speedup = baseline["avg_time_ms"] / tensorfuse["avg_time_ms"]
            
            change = current_speedup - archive_speedup
            change_str = f"{change:+.2f}x" if change != 0 else "0.00x"
            
            if change > 0:
                status = "✅ Improved"
            elif change < 0:
                status = "❌ Degraded"
            else:
                status = "➖ Same"
            
            print(f"{archive_name:<20} {archive_speedup:<10.2f} {change_str:<10} {status:<15}")

def clean_old_archives(archive_dir: str = "results/archive", keep_count: int = 10) -> None:
    """Clean old archived results, keeping only the most recent ones."""
    if not os.path.exists(archive_dir):
        return
    
    archives = [d for d in os.listdir(archive_dir) 
               if os.path.isdir(os.path.join(archive_dir, d))]
    
    if len(archives) <= keep_count:
        return
    
    # Sort by name (which includes timestamp)
    archives.sort()
    
    # Remove oldest archives
    archives_to_remove = archives[:-keep_count]
    
    for archive in archives_to_remove:
        archive_path = os.path.join(archive_dir, archive)
        try:
            shutil.rmtree(archive_path)
            print(f"🗑️  Removed old archive: {archive}")
        except Exception as e:
            print(f"❌ Failed to remove {archive}: {e}")

def generate_summary_report(results_dir: str = "results/benchmarks") -> None:
    """Generate a comprehensive summary report."""
    print("\n📄 TensorFuse Performance Report")
    print("=" * 60)
    
    # Load current results
    bert_file = os.path.join(results_dir, "bert_baseline_results.json")
    if not os.path.exists(bert_file):
        print("❌ No BERT results found")
        return
    
    with open(bert_file, 'r') as f:
        bert_data = json.load(f)
    
    config = bert_data["bert_base_benchmark"]["config"]
    results = bert_data["bert_base_benchmark"]["results"]
    
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration: BERT-base (batch={config['batch_size']}, seq={config['seq_len']})")
    print()
    
    if len(results) >= 2:
        baseline = results[0]
        tensorfuse = results[1]
        speedup = baseline["avg_time_ms"] / tensorfuse["avg_time_ms"]
        
        print("🎯 Key Performance Metrics:")
        print(f"   Current speedup: {speedup:.2f}x")
        print(f"   Target speedup: 2.0x - 7.0x")
        print(f"   Gap to target: {2.0 - speedup:.2f}x")
        print()
        
        print("📈 Performance Details:")
        print(f"   cuBLAS baseline: {baseline['avg_time_ms']:.3f}ms ({baseline['tflops']:.1f} TFLOPS)")
        print(f"   TensorFuse fused: {tensorfuse['avg_time_ms']:.3f}ms ({tensorfuse['tflops']:.1f} TFLOPS)")
        print()
        
        print("🔍 Analysis:")
        if speedup >= 2.0:
            print("   ✅ Performance target ACHIEVED!")
        else:
            print("   ❌ Performance target NOT MET")
            print(f"   💡 Need {2.0/speedup:.2f}x more performance")
            print("   📝 Optimization areas:")
            print("      - Kernel fusion efficiency")
            print("      - Memory access patterns")
            print("      - Tensor Core utilization")
            print("      - Mixed precision optimization")

def main():
    parser = argparse.ArgumentParser(description="Organize TensorFuse benchmark results")
    parser.add_argument("--copy", action="store_true",
                       help="Copy results from build directory to main results")
    parser.add_argument("--archive", action="store_true",
                       help="Archive current results with timestamp")
    parser.add_argument("--compare", action="store_true",
                       help="Compare current results with archived results")
    parser.add_argument("--clean", action="store_true",
                       help="Clean old archived results")
    parser.add_argument("--report", action="store_true",
                       help="Generate comprehensive summary report")
    parser.add_argument("--all", action="store_true",
                       help="Run all operations (copy, archive, compare, report)")
    parser.add_argument("--keep-archives", type=int, default=10,
                       help="Number of archives to keep when cleaning")
    
    args = parser.parse_args()
    
    # If no specific action, default to copy and report
    if not any([args.copy, args.archive, args.compare, args.clean, args.report, args.all]):
        args.copy = True
        args.report = True
    
    if args.all:
        args.copy = args.archive = args.compare = args.report = True
    
    if args.copy:
        print("📥 Copying results from build directory...")
        copy_results_to_main()
    
    if args.archive:
        print("\n📦 Archiving current results...")
        archive_results()
    
    if args.compare:
        print("\n📊 Comparing with archived results...")
        archived_results = load_archived_results()
        compare_results(archived_results)
    
    if args.clean:
        print(f"\n🗑️  Cleaning old archives (keeping {args.keep_archives})...")
        clean_old_archives(keep_count=args.keep_archives)
    
    if args.report:
        generate_summary_report()

if __name__ == "__main__":
    main() 