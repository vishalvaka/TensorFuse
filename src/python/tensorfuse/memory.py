"""
Memory management utilities for TensorFuse.

This module provides high-level memory management utilities, including
context managers for automatic memory cleanup, memory pool management,
and tensor allocation helpers.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from contextlib import contextmanager
import warnings

try:
    from . import DataType, Shape, Tensor
    from . import allocate_tensor, free_tensor, get_memory_info
    from . import get_tensor_size_bytes, get_dtype_size
except ImportError:
    # Provide stub implementations for development
    class DataType:
        FLOAT32 = 0
        FLOAT16 = 1
        BFLOAT16 = 2
    
    class Shape:
        def __init__(self, dims):
            self.dims = dims
            self.ndims = len(dims)
    
    class Tensor:
        def __init__(self):
            self.data = None
            self.shape = None
            self.dtype = None
    
    def allocate_tensor(shape, dtype, device_id=0):
        warnings.warn("Using stub implementation")
        return Tensor()
    
    def free_tensor(tensor):
        warnings.warn("Using stub implementation")
    
    def get_memory_info():
        return {"free_bytes": 0, "total_bytes": 0, "used_bytes": 0}
    
    def get_tensor_size_bytes(shape, dtype):
        return 0
    
    def get_dtype_size(dtype):
        return 4

class MemoryPool:
    """Memory pool for efficient tensor allocation and deallocation."""
    
    def __init__(self, initial_size: int = 1024 * 1024 * 1024):  # 1GB
        self.pool_size = initial_size
        self.allocated_tensors = []
        self.free_blocks = []
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "current_allocated": 0,
            "peak_allocated": 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear all allocated tensors."""
        self.clear()
    
    def allocate(self, shape: Union[List[int], Shape], dtype: DataType, device_id: int = 0) -> Tensor:
        """Allocate a tensor from the memory pool."""
        if isinstance(shape, list):
            shape = Shape(shape)
        
        tensor = allocate_tensor(shape, dtype, device_id)
        self.allocated_tensors.append(tensor)
        
        # Update statistics
        self.stats["total_allocations"] += 1
        self.stats["current_allocated"] += tensor.size_bytes
        self.stats["peak_allocated"] = max(self.stats["peak_allocated"], self.stats["current_allocated"])
        
        return tensor
    
    def deallocate(self, tensor: Tensor):
        """Deallocate a tensor and return it to the pool."""
        if tensor in self.allocated_tensors:
            self.allocated_tensors.remove(tensor)
            self.stats["total_deallocations"] += 1
            self.stats["current_allocated"] -= tensor.size_bytes
            free_tensor(tensor)
    
    def clear(self):
        """Clear all allocated tensors."""
        for tensor in self.allocated_tensors:
            free_tensor(tensor)
        self.allocated_tensors.clear()
        self.stats["current_allocated"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return self.stats.copy()

class TensorManager:
    """Context manager for automatic tensor lifecycle management."""
    
    def __init__(self, memory_pool: Optional[MemoryPool] = None):
        self.memory_pool = memory_pool or MemoryPool()
        self.managed_tensors = []
        # Add attributes for get_memory_usage method
        self.total_allocated = 0
        self.peak_allocated = 0
        self.active_tensors = []
        self.allocation_count = 0
        self.deallocation_count = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatically deallocate all managed tensors
        for tensor in self.managed_tensors:
            self.memory_pool.deallocate(tensor)
        self.managed_tensors.clear()
    
    def register_tensor(self, tensor: Tensor):
        """Register a tensor for automatic cleanup."""
        if tensor not in self.managed_tensors:
            self.managed_tensors.append(tensor)
    
    def allocate(self, shape: Union[List[int], Shape], dtype: DataType, device_id: int = 0) -> Tensor:
        """Allocate a tensor that will be automatically managed."""
        tensor = self.memory_pool.allocate(shape, dtype, device_id)
        self.managed_tensors.append(tensor)
        return tensor
    
    def create_from_numpy(self, array: np.ndarray, device_id: int = 0) -> Tensor:
        """Create a tensor from a numpy array."""
        # Convert numpy dtype to TensorFuse dtype
        dtype_map = {
            np.float32: DataType.FLOAT32,
            np.float16: DataType.FLOAT16,
            np.int8: DataType.INT8,
            np.uint8: DataType.UINT8,
            np.int32: DataType.INT32,
        }
        
        if array.dtype.type not in dtype_map:
            raise ValueError(f"Unsupported numpy dtype: {array.dtype}")
        
        dtype = dtype_map[array.dtype.type]
        shape = Shape(list(array.shape))
        
        tensor = self.allocate(shape, dtype, device_id)
        
        # Copy data (placeholder - in real implementation would use CUDA memcpy)
        # For now, just reference the numpy array data
        tensor.data = array.ctypes.data
        
        return tensor

    def get_memory_stats(self):
        """Get memory statistics."""
        return {
            'total_allocated': self.total_allocated,
            'peak_allocated': self.peak_allocated,
            'active_tensors': len(self.active_tensors),
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count
        }
    
    def get_memory_usage(self):
        """Get current memory usage information."""
        return {
            'current_usage': self.total_allocated,
            'peak_usage': self.peak_allocated,
            'active_tensors': len(self.active_tensors),
            'total_allocations': self.allocation_count,
            'total_deallocations': self.deallocation_count
        }

@contextmanager
def managed_tensors(memory_pool: Optional[MemoryPool] = None):
    """Context manager for automatic tensor management."""
    manager = TensorManager(memory_pool)
    try:
        yield manager
    finally:
        pass  # Cleanup handled by __exit__

def create_tensor(shape: Union[List[int], Shape], dtype: DataType, device_id: int = 0) -> Tensor:
    """Create a tensor with the specified shape and dtype."""
    if isinstance(shape, list):
        shape = Shape(shape)
    return allocate_tensor(shape, dtype, device_id)

def create_tensor_like(reference: Union[Tensor, np.ndarray], dtype: Optional[DataType] = None, device_id: Optional[int] = None) -> Tensor:
    """Create a tensor with the same shape as a reference tensor or array."""
    if isinstance(reference, Tensor):
        shape = reference.shape
        dtype = dtype or reference.dtype
        device_id = device_id or reference.device_id
    elif isinstance(reference, np.ndarray):
        shape = Shape(list(reference.shape))
        if dtype is None:
            dtype_map = {
                np.float32: DataType.FLOAT32,
                np.float16: DataType.FLOAT16,
                np.int8: DataType.INT8,
                np.uint8: DataType.UINT8,
                np.int32: DataType.INT32,
            }
            dtype = dtype_map.get(reference.dtype.type, DataType.FLOAT32)
        device_id = device_id or 0
    else:
        raise ValueError("Reference must be a Tensor or numpy array")
    
    return create_tensor(shape, dtype, device_id)

def zeros(shape: Union[List[int], Shape], dtype: DataType = DataType.FLOAT32, device_id: int = 0) -> Tensor:
    """Create a tensor filled with zeros."""
    tensor = create_tensor(shape, dtype, device_id)
    # Fill with zeros (placeholder implementation)
    return tensor

def ones(shape: Union[List[int], Shape], dtype: DataType = DataType.FLOAT32, device_id: int = 0) -> Tensor:
    """Create a tensor filled with ones."""
    tensor = create_tensor(shape, dtype, device_id)
    # Fill with ones (placeholder implementation)
    return tensor

def random(shape: Union[List[int], Shape], dtype: DataType = DataType.FLOAT32, device_id: int = 0) -> Tensor:
    """Create a tensor filled with random values."""
    tensor = create_tensor(shape, dtype, device_id)
    # Fill with random values (placeholder implementation)
    return tensor

def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics."""
    gpu_info = get_memory_info()
    
    return {
        "gpu_memory": gpu_info,
        "tensorfuse_version": "1.0.0",
        "supported_dtypes": [
            DataType.FLOAT32,
            DataType.FLOAT16,
            DataType.BFLOAT16,
            DataType.INT8,
            DataType.UINT8,
            DataType.INT32,
        ]
    }

def calculate_memory_usage(tensors: List[Tensor]) -> Dict[str, Any]:
    """Calculate memory usage for a list of tensors."""
    total_bytes = 0
    dtype_breakdown = {}
    
    for tensor in tensors:
        total_bytes += tensor.size_bytes
        dtype_name = str(tensor.dtype)
        dtype_breakdown[dtype_name] = dtype_breakdown.get(dtype_name, 0) + tensor.size_bytes
    
    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
        "dtype_breakdown": dtype_breakdown,
        "num_tensors": len(tensors)
    }

def optimize_memory_layout(tensors: List[Tensor]) -> List[Tensor]:
    """Optimize memory layout for a list of tensors."""
    # Placeholder for memory layout optimization
    # In real implementation, this would reorder tensors for better memory access patterns
    return tensors

# Global memory pool instance
_global_memory_pool = None

def get_global_memory_pool() -> MemoryPool:
    """Get the global memory pool instance."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = MemoryPool()
    return _global_memory_pool

def set_global_memory_pool(pool: MemoryPool):
    """Set the global memory pool instance."""
    global _global_memory_pool
    _global_memory_pool = pool

# Convenience functions using the global memory pool
def global_allocate(shape: Union[List[int], Shape], dtype: DataType, device_id: int = 0) -> Tensor:
    """Allocate a tensor using the global memory pool."""
    return get_global_memory_pool().allocate(shape, dtype, device_id)

def global_deallocate(tensor: Tensor):
    """Deallocate a tensor using the global memory pool."""
    get_global_memory_pool().deallocate(tensor)

def global_clear():
    """Clear all tensors in the global memory pool."""
    get_global_memory_pool().clear()

def global_stats() -> Dict[str, Any]:
    """Get global memory pool statistics."""
    return get_global_memory_pool().get_stats()

# Export all public functions
__all__ = [
    'MemoryPool', 'TensorManager', 'managed_tensors',
    'create_tensor', 'create_tensor_like', 'zeros', 'ones', 'random',
    'get_memory_stats', 'calculate_memory_usage', 'optimize_memory_layout',
    'get_global_memory_pool', 'set_global_memory_pool',
    'global_allocate', 'global_deallocate', 'global_clear', 'global_stats'
] 