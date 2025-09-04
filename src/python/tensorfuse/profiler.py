"""Profiling utilities for TensorFuse."""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

class SimpleProfiler:
    """Simple profiler for timing operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.operation_times = {}
        self.current_operation = None
        self.operation_starts = {}  # Track start times for multiple operations
    
    def start(self, operation_name: Optional[str] = None):
        """Start timing for an operation."""
        if operation_name is None:
            # Global timing
            self.start_time = time.time()
        else:
            # Operation-specific timing
            self.operation_starts[operation_name] = time.time()
    
    def stop(self, operation_name: Optional[str] = None):
        """Stop timing and return elapsed time."""
        if operation_name is None:
            # Global timing
            if self.start_time is None:
                return 0.0
            elapsed = time.time() - self.start_time
            self.start_time = None
            return elapsed
        else:
            # Operation-specific timing
            if operation_name not in self.operation_starts:
                return 0.0
            start_time = self.operation_starts[operation_name]
            elapsed = time.time() - start_time
            
            # Store the result
            self.operation_times[operation_name] = elapsed
            self.metrics[operation_name] = {
                'execution_time': elapsed,
                'total_time': elapsed,
                'timestamp': start_time
            }
            
            del self.operation_starts[operation_name]
            return elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get profiler metrics."""
        return self.metrics.copy()
    
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results."""
        return self.metrics.copy()
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling a specific operation."""
        self.current_operation = operation_name
        start_time = time.time()
        
        try:
            yield self
        finally:
            elapsed = time.time() - start_time
            self.operation_times[operation_name] = elapsed
            self.metrics[operation_name] = {
                'execution_time': elapsed,
                'total_time': elapsed,
                'timestamp': start_time
            }
            self.current_operation = None

@contextmanager
def profile(name: str = "operation"):
    """Context manager for profiling operations."""
    profiler = SimpleProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        elapsed = profiler.stop()
        print(f"{name}: {elapsed:.4f}s")

__all__ = ['SimpleProfiler', 'profile'] 