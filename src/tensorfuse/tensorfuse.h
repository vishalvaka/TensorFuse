#pragma once

/**
 * @file tensorfuse.h
 * @brief Main TensorFuse API header
 * 
 * TensorFuse - Tensor-Core-Optimized Transformer Inference Library
 * 
 * This library provides drop-in replacement for standard transformer operations
 * with 2-7x performance improvements over cuBLASLt through kernel fusion and
 * Tensor Core optimization.
 */

#include "types.h"
#include "config.h"
#include "profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize TensorFuse library
 * @param config Configuration parameters
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_init(const TensorFuseConfig* config);

/**
 * @brief Shutdown TensorFuse library
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_shutdown();

/**
 * @brief Get library version
 * @return Version string
 */
const char* tensorfuse_get_version();

/**
 * @brief Get error string for status code
 * @param status Status code
 * @return Error message string
 */
const char* tensorfuse_get_error_string(TensorFuseStatus status);

// ==============================================================================
// Fused Operations
// ==============================================================================

/**
 * @brief Fused GEMM + Bias + GELU operation
 * 
 * Performs: output = GELU(input @ weight + bias)
 * 
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param weight Weight tensor [hidden_dim, ffn_dim]
 * @param bias Bias tensor [ffn_dim]
 * @param output Output tensor [batch, seq_len, ffn_dim]
 * @param stream CUDA stream for async execution
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_fused_gemm_bias_gelu(
    const TensorFuseTensor* input,
    const TensorFuseTensor* weight,
    const TensorFuseTensor* bias,
    TensorFuseTensor* output,
    cudaStream_t stream
);

/**
 * @brief Fused Softmax + Dropout operation
 * 
 * Performs: output = Dropout(Softmax(input, axis), p)
 * 
 * @param input Input tensor [batch, heads, seq_len, seq_len]
 * @param output Output tensor [batch, heads, seq_len, seq_len]
 * @param dropout_mask Dropout mask output [batch, heads, seq_len, seq_len]
 * @param dropout_prob Dropout probability (0.0 to 1.0)
 * @param stream CUDA stream for async execution
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_fused_softmax_dropout(
    const TensorFuseTensor* input,
    TensorFuseTensor* output,
    unsigned char* dropout_mask,
    float dropout_prob,
    cudaStream_t stream
);

/**
 * @brief Fused Multi-Head Attention operation
 * 
 * Performs complete multi-head attention with fused operations
 * 
 * @param query Query tensor [batch, seq_len, hidden_dim]
 * @param key Key tensor [batch, seq_len, hidden_dim]
 * @param value Value tensor [batch, seq_len, hidden_dim]
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param num_heads Number of attention heads
 * @param dropout_prob Dropout probability for attention weights
 * @param stream CUDA stream for async execution
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_fused_multi_head_attention(
    const TensorFuseTensor* query,
    const TensorFuseTensor* key,
    const TensorFuseTensor* value,
    TensorFuseTensor* output,
    int num_heads,
    float dropout_prob,
    cudaStream_t stream
);

/**
 * @brief Fused Transformer Layer operation
 * 
 * Performs complete transformer layer with optimized kernel fusion
 * 
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param params Transformer layer parameters
 * @param stream CUDA stream for async execution
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_fused_transformer_layer(
    const TensorFuseTensor* input,
    TensorFuseTensor* output,
    const TensorFuseTransformerParams* params,
    cudaStream_t stream
);

// ==============================================================================
// Autotuner Interface
// ==============================================================================

/**
 * @brief Auto-tune kernels for specific hardware and model configuration
 * 
 * @param model_config Model configuration parameters
 * @param output_path Path to save tuning results (JSON format)
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_autotune(
    const TensorFuseModelConfig* model_config,
    const char* output_path
);

/**
 * @brief Load pre-tuned kernel configurations
 * 
 * @param config_path Path to tuning configuration file
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_load_tuned_config(const char* config_path);

// ==============================================================================
// Profiling Interface
// ==============================================================================

/**
 * @brief Start profiling session
 * 
 * @param profile_config Profiling configuration
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_start_profiling(const TensorFuseProfileConfig* profile_config);

/**
 * @brief Stop profiling session and save results
 * 
 * @param output_path Path to save profiling results
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_stop_profiling(const char* output_path);

// ==============================================================================
// Memory Management
// ==============================================================================

/**
 * @brief Allocate tensor with optimal memory alignment
 * 
 * @param tensor Tensor to allocate
 * @param shape Tensor shape
 * @param dtype Data type
 * @param device_id CUDA device ID
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_allocate_tensor(
    TensorFuseTensor* tensor,
    const TensorFuseShape* shape,
    TensorFuseDataType dtype,
    int device_id
);

/**
 * @brief Free tensor memory
 * 
 * @param tensor Tensor to free
 * @return TENSORFUSE_SUCCESS on success, error code otherwise
 */
TensorFuseStatus tensorfuse_free_tensor(TensorFuseTensor* tensor);

#ifdef __cplusplus
}
#endif 