"""
Core TensorFuse Python utilities and high-level wrapper functions.

This module provides high-level Python interfaces for TensorFuse operations,
including PyTorch integration, convenient tensor operations, and performance
optimizations.
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, TYPE_CHECKING
import warnings

# Type checking imports
if TYPE_CHECKING:
    import torch

class TensorFuseOp:
    """Base class for TensorFuse operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = None
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_metrics(self):
        """Get performance metrics from last operation."""
        return self.metrics

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    class FusedGemmBiasGelu(nn.Module):
        """Fused GEMM + Bias + GELU operation wrapper."""
        
        def __init__(self, m: Optional[int] = None, n: Optional[int] = None, k: Optional[int] = None, 
                     dtype: Optional[Any] = None, enable_profiling: bool = False):
            """
            Initialize FusedGemmBiasGelu operation.
            
            Args:
                m: Number of rows in input matrix (batch * seq_len)
                n: Number of columns in weight matrix (output dimension)
                k: Number of columns in input matrix / rows in weight matrix (hidden dimension)
                dtype: Data type for the operation (FP32, FP16, etc.)
                enable_profiling: Enable profiling for this operation
            """
            super().__init__()
            self.m = m
            self.n = n
            self.k = k
            self.dtype = dtype
            self.enable_profiling = enable_profiling
            self.metrics = None
            
            # Add dummy parameters for PyTorch compatibility
            if m and n and k:
                self.weight = nn.Parameter(torch.randn(k, n))
                self.bias = nn.Parameter(torch.randn(n))
        
        def forward(self, 
                   input_tensor,
                   weight: Optional[Any] = None,
                   bias: Optional[Any] = None):
            """Forward pass for PyTorch compatibility."""
            if weight is None:
                weight = self.weight.detach().numpy() if hasattr(self, 'weight') else None
            if bias is None:
                bias = self.bias.detach().numpy() if hasattr(self, 'bias') else None
            
            return self.__call__(input_tensor, weight, bias)
        
        def __call__(self, 
                     input_tensor,
                     weight,
                     bias,
                     output: Optional[Any] = None,
                     stream=None):
            """
            Perform fused GEMM + Bias + GELU operation.
            
            Args:
                input_tensor: Input tensor [batch, seq_len, hidden_dim]
                weight: Weight tensor [hidden_dim, ffn_dim]
                bias: Bias tensor [ffn_dim]
                output: Output tensor (optional)
                stream: CUDA stream for async execution
                
            Returns:
                Output tensor [batch, seq_len, ffn_dim]
            """
            # Handle PyTorch tensors
            input_is_torch = hasattr(input_tensor, 'device')
            if input_is_torch:
                device = str(input_tensor.device)
                input_np = convert_torch_to_numpy(input_tensor)
                weight_np = convert_torch_to_numpy(weight)
                bias_np = convert_torch_to_numpy(bias)
            else:
                input_np = input_tensor
                weight_np = weight
                bias_np = bias
            
            # Create output tensor if not provided
            if output is None:
                batch_size, seq_len, hidden_dim = input_np.shape
                ffn_dim = weight_np.shape[1]
                output_np = np.zeros((batch_size, seq_len, ffn_dim), dtype=input_np.dtype)
            else:
                output_np = convert_torch_to_numpy(output) if input_is_torch else output
            
            # Import the actual function
            from . import fused_gemm_bias_gelu
            
            # Perform the operation
            fused_gemm_bias_gelu(input_np, weight_np, bias_np, output_np, stream)
            
            # Convert back to PyTorch if needed
            if input_is_torch:
                return convert_numpy_to_torch(output_np, device)
            else:
                return output_np

except ImportError:
    TORCH_AVAILABLE = False
    
    class FusedGemmBiasGelu(TensorFuseOp):
        """Fused GEMM + Bias + GELU operation wrapper."""
        
        def __init__(self, m: Optional[int] = None, n: Optional[int] = None, k: Optional[int] = None, 
                     dtype: Optional[Any] = None, enable_profiling: bool = False):
            """
            Initialize FusedGemmBiasGelu operation.
            
            Args:
                m: Number of rows in input matrix (batch * seq_len)
                n: Number of columns in weight matrix (output dimension)
                k: Number of columns in input matrix / rows in weight matrix (hidden dimension)
                dtype: Data type for the operation (FP32, FP16, etc.)
                enable_profiling: Enable profiling for this operation
            """
            super().__init__("fused_gemm_bias_gelu")
            self.m = m
            self.n = n
            self.k = k
            self.dtype = dtype
            self.enable_profiling = enable_profiling
            self.metrics = None
        
        def __call__(self, 
                     input_tensor,
                     weight,
                     bias,
                     output: Optional[Any] = None,
                     stream=None):
            """
            Perform fused GEMM + Bias + GELU operation.
            
            Args:
                input_tensor: Input tensor [batch, seq_len, hidden_dim]
                weight: Weight tensor [hidden_dim, ffn_dim]
                bias: Bias tensor [ffn_dim]
                output: Output tensor (optional)
                stream: CUDA stream for async execution
                
            Returns:
                Output tensor [batch, seq_len, ffn_dim]
            """
            # Create output tensor if not provided
            if output is None:
                batch_size, seq_len, hidden_dim = input_tensor.shape
                ffn_dim = weight.shape[1]
                output = np.zeros((batch_size, seq_len, ffn_dim), dtype=input_tensor.dtype)
            
            # Import the actual function
            from . import fused_gemm_bias_gelu
            
            # Perform the operation
            fused_gemm_bias_gelu(input_tensor, weight, bias, output, stream)
            
            return output

def convert_torch_to_numpy(tensor: Any) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch not available")
    
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def convert_numpy_to_torch(array: np.ndarray, device: str = 'cuda') -> Any:
    """Convert numpy array to PyTorch tensor."""
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch not available")
    
    tensor = torch.from_numpy(array)
    if device != 'cpu':
        tensor = tensor.to(device)
    return tensor

class FusedSoftmaxDropout(TensorFuseOp):
    """Fused Softmax + Dropout operation wrapper."""
    
    def __init__(self, enable_profiling: bool = False):
        super().__init__("fused_softmax_dropout")
        self.enable_profiling = enable_profiling
    
    def __call__(self,
                 input_tensor: Union[np.ndarray, 'torch.Tensor'],
                 dropout_prob: float = 0.1,
                 output: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
                 dropout_mask: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
                 stream=None) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Perform fused Softmax + Dropout operation.
        
        Args:
            input_tensor: Input tensor [batch, heads, seq_len, seq_len]
            dropout_prob: Dropout probability
            output: Output tensor (optional)
            dropout_mask: Dropout mask (optional)
            stream: CUDA stream for async execution
            
        Returns:
            Tuple of (output, dropout_mask)
        """
        # Handle PyTorch tensors
        input_is_torch = TORCH_AVAILABLE and isinstance(input_tensor, 'torch.Tensor')
        if input_is_torch:
            device = input_tensor.device
            input_np = convert_torch_to_numpy(input_tensor)
        else:
            input_np = input_tensor
        
        # Create output and mask tensors if not provided
        if output is None:
            output_np = np.zeros_like(input_np)
        else:
            output_np = convert_torch_to_numpy(output) if input_is_torch else output
        
        if dropout_mask is None:
            mask_np = np.zeros(input_np.shape, dtype=np.uint8)
        else:
            mask_np = convert_torch_to_numpy(dropout_mask) if input_is_torch else dropout_mask
        
        # Import the actual function
        from . import fused_softmax_dropout
        
        # Perform the operation
        fused_softmax_dropout(input_np, output_np, mask_np, dropout_prob, stream)
        
        # Convert back to PyTorch if needed
        if input_is_torch:
            return (convert_numpy_to_torch(output_np, device), 
                    convert_numpy_to_torch(mask_np, device))
        else:
            return output_np, mask_np

class FusedMultiHeadAttention(TensorFuseOp):
    """Fused Multi-Head Attention operation wrapper."""
    
    def __init__(self, num_heads: int, enable_profiling: bool = False):
        super().__init__("fused_multi_head_attention")
        self.num_heads = num_heads
        self.enable_profiling = enable_profiling
    
    def __call__(self,
                 query: Union[np.ndarray, 'torch.Tensor'],
                 key: Union[np.ndarray, 'torch.Tensor'],
                 value: Union[np.ndarray, 'torch.Tensor'],
                 dropout_prob: float = 0.1,
                 output: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
                 stream=None) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Perform fused Multi-Head Attention operation.
        
        Args:
            query: Query tensor [batch, seq_len, hidden_dim]
            key: Key tensor [batch, seq_len, hidden_dim]
            value: Value tensor [batch, seq_len, hidden_dim]
            dropout_prob: Dropout probability for attention weights
            output: Output tensor (optional)
            stream: CUDA stream for async execution
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Handle PyTorch tensors
        input_is_torch = TORCH_AVAILABLE and isinstance(query, 'torch.Tensor')
        if input_is_torch:
            device = query.device
            query_np = convert_torch_to_numpy(query)
            key_np = convert_torch_to_numpy(key)
            value_np = convert_torch_to_numpy(value)
        else:
            query_np = query
            key_np = key
            value_np = value
        
        # Create output tensor if not provided
        if output is None:
            output_np = np.zeros_like(query_np)
        else:
            output_np = convert_torch_to_numpy(output) if input_is_torch else output
        
        # Import the actual function
        from . import fused_multi_head_attention
        
        # Perform the operation
        fused_multi_head_attention(query_np, key_np, value_np, output_np, 
                                   self.num_heads, dropout_prob, stream)
        
        # Convert back to PyTorch if needed
        if input_is_torch:
            return convert_numpy_to_torch(output_np, device)
        else:
            return output_np

# High-level convenience functions
def gemm_bias_gelu(input_tensor, weight, bias, output=None, stream=None):
    """Convenience function for fused GEMM + Bias + GELU."""
    op = FusedGemmBiasGelu()
    return op(input_tensor, weight, bias, output, stream)

def softmax_dropout(input_tensor, dropout_prob=0.1, output=None, dropout_mask=None, stream=None):
    """Convenience function for fused Softmax + Dropout."""
    op = FusedSoftmaxDropout()
    return op(input_tensor, dropout_prob, output, dropout_mask, stream)

def multi_head_attention(query, key, value, num_heads, dropout_prob=0.1, output=None, stream=None):
    """Convenience function for fused Multi-Head Attention."""
    op = FusedMultiHeadAttention(num_heads)
    return op(query, key, value, dropout_prob, output, stream)

# PyTorch integration
if TORCH_AVAILABLE:
    class TensorFuseFunction(torch.autograd.Function):
        """PyTorch autograd function for TensorFuse operations."""
        
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, op_name):
            ctx.save_for_backward(input_tensor, weight, bias)
            ctx.op_name = op_name
            
            if op_name == "fused_gemm_bias_gelu":
                return gemm_bias_gelu(input_tensor, weight, bias)
            else:
                raise ValueError(f"Unknown operation: {op_name}")
        
        @staticmethod
        def backward(ctx, grad_output):
            # Placeholder for backward pass
            input_tensor, weight, bias = ctx.saved_tensors
            return grad_output, None, None, None
    
    def tensorfuse_gemm_bias_gelu(input_tensor, weight, bias):
        """PyTorch autograd-compatible fused GEMM + Bias + GELU."""
        return TensorFuseFunction.apply(input_tensor, weight, bias, "fused_gemm_bias_gelu")

# Export all public functions
__all__ = [
    'TensorFuseOp', 'FusedGemmBiasGelu', 'FusedSoftmaxDropout', 'FusedMultiHeadAttention',
    'gemm_bias_gelu', 'softmax_dropout', 'multi_head_attention',
    'convert_torch_to_numpy', 'convert_numpy_to_torch',
]

if TORCH_AVAILABLE:
    __all__.extend(['TensorFuseFunction', 'tensorfuse_gemm_bias_gelu']) 