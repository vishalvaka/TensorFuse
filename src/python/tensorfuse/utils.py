"""Utility functions for TensorFuse."""

import numpy as np
from typing import Any, Dict, List, Optional, Union

def validate_tensor_shape(shape: Union[List[int], tuple]) -> List[int]:
    """Validate and normalize tensor shape."""
    if not isinstance(shape, (list, tuple)):
        raise ValueError("Shape must be a list or tuple")
    
    shape = list(shape)
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("All dimensions must be positive integers")
    
    return shape

def validate_shape(shape: Union[List[int], tuple]) -> bool:
    """Validate tensor shape and return boolean (alias for test compatibility)."""
    try:
        if not isinstance(shape, (list, tuple)):
            return False
        if len(shape) == 0:
            return False
        shape = list(shape)
        for dim in shape:
            if not isinstance(dim, int) or dim <= 0:
                return False
        return True
    except:
        return False

def get_dtype_size(dtype) -> int:
    """Get size of data type in bytes."""
    # Handle both enum and integer values
    if hasattr(dtype, 'value'):
        dtype_val = dtype.value
    else:
        dtype_val = dtype
    
    # Map to sizes based on TensorFuse data types
    size_map = {
        0: 4,  # FLOAT32
        1: 2,  # FLOAT16
        2: 2,  # BFLOAT16
        3: 1,  # INT8
        4: 1,  # UINT8
        5: 4,  # INT32
        6: 1,  # FP8_E4M3
        7: 1,  # FP8_E5M2
    }
    
    return size_map.get(dtype_val, 4)  # Default to 4 bytes

def calculate_tensor_size(shape: Union[List[int], tuple], dtype) -> int:
    """Calculate tensor size in bytes."""
    shape = validate_tensor_shape(shape)
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    # Handle DataType enum or integer
    dtype_size = get_dtype_size(dtype)
    return total_elements * dtype_size

def dtype_from_string(dtype_str: str):
    """Convert string to DataType enum."""
    try:
        # Import DataType from parent module
        from . import DataType
        
        # Map string names to DataType enum values (only include available ones)
        dtype_map = {
            'FP32': DataType.FLOAT32,
            'FLOAT32': DataType.FLOAT32,
            'FP16': DataType.FLOAT16,
            'FLOAT16': DataType.FLOAT16,
            'BF16': DataType.BFLOAT16,
            'BFLOAT16': DataType.BFLOAT16,
        }
        
        # Add optional types only if they exist in the enum
        int8_type = getattr(DataType, 'INT8', None)
        if int8_type is not None:
            dtype_map['INT8'] = int8_type
        
        uint8_type = getattr(DataType, 'UINT8', None)
        if uint8_type is not None:
            dtype_map['UINT8'] = uint8_type
            
        int32_type = getattr(DataType, 'INT32', None)
        if int32_type is not None:
            dtype_map['INT32'] = int32_type
            
        fp8_e4m3_type = getattr(DataType, 'FP8_E4M3', None)
        if fp8_e4m3_type is not None:
            dtype_map['FP8_E4M3'] = fp8_e4m3_type
            
        fp8_e5m2_type = getattr(DataType, 'FP8_E5M2', None)
        if fp8_e5m2_type is not None:
            dtype_map['FP8_E5M2'] = fp8_e5m2_type
        
        if dtype_str in dtype_map:
            return dtype_map[dtype_str]
        else:
            raise ValueError(f"Unknown data type string: {dtype_str}")
            
    except ImportError:
        # Fallback for when module not available
        raise ValueError(f"DataType module not available, cannot convert: {dtype_str}")

def numpy_dtype_to_tensorfuse(np_dtype):
    """Convert numpy dtype to TensorFuse DataType."""
    try:
        from . import DataType
        
        # Map numpy dtypes to TensorFuse DataType (only include available ones)
        dtype_map = {
            np.float32: DataType.FLOAT32,
            np.float16: DataType.FLOAT16,
        }
        
        # Add optional types only if they exist in the enum
        int8_type = getattr(DataType, 'INT8', None)
        if int8_type is not None:
            dtype_map[np.int8] = int8_type
            
        uint8_type = getattr(DataType, 'UINT8', None)
        if uint8_type is not None:
            dtype_map[np.uint8] = uint8_type
            
        int32_type = getattr(DataType, 'INT32', None)
        if int32_type is not None:
            dtype_map[np.int32] = int32_type
        
        # Handle numpy dtype objects
        if hasattr(np_dtype, 'type'):
            np_type = np_dtype.type
        else:
            np_type = np_dtype
            
        if np_type in dtype_map:
            return dtype_map[np_type]
        else:
            raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
            
    except ImportError:
        raise ValueError(f"DataType module not available, cannot convert: {np_dtype}")

def is_shape_compatible(shape: Union[List[int], tuple]) -> bool:
    """Check if shape is compatible with TensorFuse."""
    try:
        if not isinstance(shape, (list, tuple)):
            return False
        if len(shape) == 0:
            return False
        if len(shape) > 8:  # TENSORFUSE_MAX_DIMS
            return False
        
        shape = list(shape)
        for dim in shape:
            if not isinstance(dim, int) or dim <= 0:
                return False
        
        return True
    except:
        return False

def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.1f} MB"
    else:
        return f"{bytes_val / (1024**3):.1f} GB"

def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    return {
        "name": "Default Device",
        "compute_capability": "Unknown",
        "memory_total": 0,
        "memory_free": 0,
    }

def dtype_to_string(dtype) -> str:
    """Convert DataType enum to string representation."""
    try:
        # Map DataType enum values to strings (using FP format expected by tests)
        dtype_map = {
            0: 'FP32',       # DataType.FLOAT32
            1: 'FP16',       # DataType.FLOAT16
            2: 'BF16',       # DataType.BFLOAT16
            3: 'INT8',       # DataType.INT8 if available
            4: 'UINT8',      # DataType.UINT8 if available
            5: 'INT32',      # DataType.INT32 if available
        }
        
        # Handle both enum values and integer values
        if hasattr(dtype, 'value'):
            dtype_value = dtype.value
        else:
            dtype_value = int(dtype)
            
        return dtype_map.get(dtype_value, f'UNKNOWN_DTYPE_{dtype_value}')
        
    except Exception as e:
        return f'UNKNOWN_DTYPE_{dtype}'

def validate_dtype(dtype) -> bool:
    """Validate that a data type is supported by TensorFuse."""
    try:
        # Import DataType from parent module
        from . import DataType
        
        # Handle string inputs by converting to enum
        if isinstance(dtype, str):
            dtype_from_string_func = globals().get('dtype_from_string')
            if dtype_from_string_func:
                try:
                    dtype = dtype_from_string_func(dtype)
                except (ValueError, KeyError):
                    return False
            else:
                # Fallback: manual string mapping
                string_to_dtype = {
                    'FP32': DataType.FLOAT32,
                    'FP16': DataType.FLOAT16,
                    'BFLOAT16': DataType.BFLOAT16,
                    'BF16': DataType.BFLOAT16,
                    'FLOAT32': DataType.FLOAT32,
                    'FLOAT16': DataType.FLOAT16,
                }
                dtype = string_to_dtype.get(dtype.upper())
                if dtype is None:
                    return False
        
        # Check if it's a valid DataType enum value
        valid_dtypes = [
            DataType.FLOAT32,
            DataType.FLOAT16,
            DataType.BFLOAT16,
        ]
        
        # Add optional types if available
        int8_type = getattr(DataType, 'INT8', None)
        if int8_type is not None:
            valid_dtypes.append(int8_type)
            
        uint8_type = getattr(DataType, 'UINT8', None)
        if uint8_type is not None:
            valid_dtypes.append(uint8_type)
            
        int32_type = getattr(DataType, 'INT32', None)
        if int32_type is not None:
            valid_dtypes.append(int32_type)
        
        return dtype in valid_dtypes
        
    except Exception:
        return False

def validate_dimensions(dims) -> bool:
    """Validate that dimensions are within supported limits."""
    try:
        # Import TENSORFUSE_MAX_DIMS from parent module
        from . import TENSORFUSE_MAX_DIMS
        max_dims = TENSORFUSE_MAX_DIMS
    except ImportError:
        max_dims = 8  # Default fallback
    
    if not isinstance(dims, (list, tuple)):
        return False
    
    # For practical use, limit to 4 dimensions (batch, height, width, channels) 
    # even though the library technically supports more
    practical_max_dims = 4
    
    # Check that dimensions don't exceed practical maximum
    if len(dims) > practical_max_dims:
        return False
    
    # Check that all dimensions are positive integers
    for dim in dims:
        if not isinstance(dim, int) or dim <= 0:
            return False
    
    return True

__all__ = [
    'validate_tensor_shape', 'validate_shape', 'calculate_tensor_size', 
    'format_bytes', 'get_device_info', 'dtype_from_string', 
    'numpy_dtype_to_tensorfuse', 'is_shape_compatible', 'get_dtype_size',
    'dtype_to_string', 'validate_dtype', 'validate_dimensions'
] 