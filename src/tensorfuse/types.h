#pragma once

/**
 * @file types.h
 * @brief Core type definitions for TensorFuse library
 */

#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
#include <string>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// Status Codes
// ==============================================================================

/**
 * @brief Status codes for TensorFuse operations
 */
typedef enum {
    TENSORFUSE_SUCCESS = 0,
    TENSORFUSE_ERROR_INVALID_ARGUMENT = 1,
    TENSORFUSE_ERROR_CUDA_ERROR = 2,
    TENSORFUSE_ERROR_OUT_OF_MEMORY = 3,
    TENSORFUSE_ERROR_NOT_INITIALIZED = 4,
    TENSORFUSE_ERROR_INVALID_CONFIGURATION = 5,
    TENSORFUSE_ERROR_KERNEL_LAUNCH_FAILED = 6,
    TENSORFUSE_ERROR_AUTOTUNER_FAILED = 7,
    TENSORFUSE_ERROR_PROFILING_FAILED = 8,
    TENSORFUSE_ERROR_FILE_IO = 9,
    TENSORFUSE_ERROR_UNSUPPORTED_OPERATION = 10,
    TENSORFUSE_ERROR_INVALID_TENSOR_SHAPE = 11,
    TENSORFUSE_ERROR_UNKNOWN = 99
} TensorFuseStatus;

// ==============================================================================
// Data Types
// ==============================================================================

/**
 * @brief Supported data types for tensors
 */
typedef enum {
    TENSORFUSE_FLOAT32 = 0,    ///< 32-bit floating point
    TENSORFUSE_FLOAT16 = 1,    ///< 16-bit floating point (half precision)
    TENSORFUSE_BFLOAT16 = 2,   ///< 16-bit brain floating point
    TENSORFUSE_INT8 = 3,       ///< 8-bit signed integer
    TENSORFUSE_UINT8 = 4,      ///< 8-bit unsigned integer
    TENSORFUSE_INT32 = 5,      ///< 32-bit signed integer
    TENSORFUSE_FP8_E4M3 = 6,   ///< 8-bit floating point E4M3 (Hopper+)
    TENSORFUSE_FP8_E5M2 = 7    ///< 8-bit floating point E5M2 (Hopper+)
} TensorFuseDataType;

/**
 * @brief Tensor layout formats
 */
typedef enum {
    TENSORFUSE_LAYOUT_ROW_MAJOR = 0,    ///< Row-major (C-style) layout
    TENSORFUSE_LAYOUT_COL_MAJOR = 1,    ///< Column-major (Fortran-style) layout
    TENSORFUSE_LAYOUT_NCHW = 2,         ///< Batch-Channel-Height-Width
    TENSORFUSE_LAYOUT_NHWC = 3,         ///< Batch-Height-Width-Channel
    TENSORFUSE_LAYOUT_CUSTOM = 4        ///< Custom layout with strides
} TensorFuseLayout;

/**
 * @brief Activation functions
 */
typedef enum {
    TENSORFUSE_ACTIVATION_NONE = 0,     ///< No activation
    TENSORFUSE_ACTIVATION_RELU = 1,     ///< ReLU activation
    TENSORFUSE_ACTIVATION_GELU = 2,     ///< GELU activation
    TENSORFUSE_ACTIVATION_SILU = 3,     ///< SiLU (Swish) activation
    TENSORFUSE_ACTIVATION_TANH = 4,     ///< Tanh activation
    TENSORFUSE_ACTIVATION_SIGMOID = 5   ///< Sigmoid activation
} TensorFuseActivation;

// ==============================================================================
// Tensor Shape and Descriptor
// ==============================================================================

/**
 * @brief Maximum number of tensor dimensions
 */
#define TENSORFUSE_MAX_DIMS 8

/**
 * @brief Tensor shape descriptor
 */
typedef struct {
    int ndims;                           ///< Number of dimensions
    int dims[TENSORFUSE_MAX_DIMS];       ///< Dimension sizes
    int strides[TENSORFUSE_MAX_DIMS];    ///< Stride in elements for each dimension
} TensorFuseShape;

/**
 * @brief Tensor descriptor
 */
typedef struct {
    void* data;                          ///< Pointer to tensor data
    TensorFuseShape shape;               ///< Tensor shape
    TensorFuseDataType dtype;            ///< Data type
    TensorFuseLayout layout;             ///< Memory layout
    int device_id;                       ///< CUDA device ID
    size_t size_bytes;                   ///< Total size in bytes
    bool is_contiguous;                  ///< Whether tensor is contiguous
    void* scale;                         ///< Scale factor for quantized types (optional)
    void* zero_point;                    ///< Zero point for quantized types (optional)
} TensorFuseTensor;

// ==============================================================================
// Configuration Structures
// ==============================================================================

/**
 * @brief Library configuration
 */
typedef struct {
    int device_count;                    ///< Number of CUDA devices to use
    int* device_ids;                     ///< Array of device IDs
    size_t workspace_size_bytes;         ///< Workspace memory size per device
    bool enable_profiling;               ///< Enable profiling support
    bool enable_autotuning;              ///< Enable automatic kernel tuning
    bool enable_fp8;                     ///< Enable FP8 support (Hopper+)
    int log_level;                       ///< Logging level (0=quiet, 1=info, 2=debug)
    const char* cache_dir;               ///< Directory for kernel cache files
} TensorFuseConfig;

/**
 * @brief Model configuration for autotuning
 */
typedef struct {
    int batch_size;                      ///< Batch size
    int seq_length;                      ///< Sequence length
    int hidden_dim;                      ///< Hidden dimension
    int num_heads;                       ///< Number of attention heads
    int ffn_dim;                         ///< FFN intermediate dimension
    int num_layers;                      ///< Number of transformer layers
    TensorFuseDataType dtype;            ///< Primary data type
    bool use_flash_attention;            ///< Use Flash Attention optimization
    bool use_mixed_precision;            ///< Use mixed precision
    const char* model_name;              ///< Model name for caching
} TensorFuseModelConfig;

/**
 * @brief Transformer layer parameters
 */
typedef struct {
    // Attention weights
    TensorFuseTensor* query_weight;      ///< Query projection weight
    TensorFuseTensor* key_weight;        ///< Key projection weight
    TensorFuseTensor* value_weight;      ///< Value projection weight
    TensorFuseTensor* output_weight;     ///< Output projection weight
    
    // Attention biases
    TensorFuseTensor* query_bias;        ///< Query projection bias
    TensorFuseTensor* key_bias;          ///< Key projection bias
    TensorFuseTensor* value_bias;        ///< Value projection bias
    TensorFuseTensor* output_bias;       ///< Output projection bias
    
    // FFN weights
    TensorFuseTensor* ffn_weight1;       ///< First FFN layer weight
    TensorFuseTensor* ffn_weight2;       ///< Second FFN layer weight
    TensorFuseTensor* ffn_bias1;         ///< First FFN layer bias
    TensorFuseTensor* ffn_bias2;         ///< Second FFN layer bias
    
    // Layer normalization
    TensorFuseTensor* ln1_weight;        ///< First layer norm weight
    TensorFuseTensor* ln1_bias;          ///< First layer norm bias
    TensorFuseTensor* ln2_weight;        ///< Second layer norm weight
    TensorFuseTensor* ln2_bias;          ///< Second layer norm bias
    
    // Configuration
    int num_heads;                       ///< Number of attention heads
    float dropout_prob;                  ///< Dropout probability
    TensorFuseActivation ffn_activation; ///< FFN activation function
    float layer_norm_eps;                ///< Layer normalization epsilon
    bool use_bias;                       ///< Whether to use bias terms
    bool use_flash_attention;            ///< Use Flash Attention
} TensorFuseTransformerParams;

// ==============================================================================
// Profiling Structures
// ==============================================================================

/**
 * @brief Profiling configuration
 */
typedef struct {
    bool enable_nsight_systems;          ///< Enable Nsight Systems profiling
    bool enable_nsight_compute;          ///< Enable Nsight Compute profiling
    bool enable_nvtx;                    ///< Enable NVTX markers
    bool enable_cuda_events;             ///< Enable CUDA event timing
    bool enable_memory_tracking;         ///< Enable memory usage tracking
    const char* output_prefix;           ///< Prefix for output files
    int warmup_iterations;               ///< Warmup iterations before profiling
    int profile_iterations;              ///< Number of profiling iterations
} TensorFuseProfileConfig;

/**
 * @brief Performance metrics
 */
typedef struct {
    float kernel_time_ms;                ///< Kernel execution time (ms)
    float memory_bandwidth_gbps;         ///< Memory bandwidth (GB/s)
    float tensor_core_utilization;       ///< Tensor Core utilization (%)
    float flops_per_second;              ///< FLOPS per second
    size_t memory_used_bytes;            ///< Memory used (bytes)
    size_t memory_peak_bytes;            ///< Peak memory usage (bytes)
    int num_kernel_launches;             ///< Number of kernel launches
    float cpu_time_ms;                   ///< CPU time (ms)
    float total_time_ms;                 ///< Total time (ms)
} TensorFuseMetrics;

// ==============================================================================
// Kernel Configuration
// ==============================================================================

/**
 * @brief Kernel tile configuration
 */
typedef struct {
    int tile_m;                          ///< Tile size in M dimension
    int tile_n;                          ///< Tile size in N dimension
    int tile_k;                          ///< Tile size in K dimension
    int warp_m;                          ///< Warp tile size in M dimension
    int warp_n;                          ///< Warp tile size in N dimension
    int warp_k;                          ///< Warp tile size in K dimension
    int stages;                          ///< Number of pipeline stages
    int split_k;                         ///< Split-K factor
} TensorFuseKernelConfig;

/**
 * @brief Autotuner configuration
 */
typedef struct {
    int max_iterations;                  ///< Maximum tuning iterations
    float time_limit_seconds;            ///< Time limit for tuning
    int num_warmup_runs;                 ///< Warmup runs per configuration
    int num_timing_runs;                 ///< Timing runs per configuration
    float relative_threshold;            ///< Relative performance threshold
    bool use_heuristics;                 ///< Use performance heuristics
    bool cache_results;                  ///< Cache tuning results
    const char* cache_file;              ///< Cache file path
} TensorFuseAutotunerConfig;

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * @brief Get size of data type in bytes
 * @param dtype Data type
 * @return Size in bytes
 */
static inline size_t tensorfuse_sizeof_dtype(TensorFuseDataType dtype) {
    switch (dtype) {
        case TENSORFUSE_FLOAT32: return 4;
        case TENSORFUSE_FLOAT16: return 2;
        case TENSORFUSE_BFLOAT16: return 2;
        case TENSORFUSE_INT8: return 1;
        case TENSORFUSE_UINT8: return 1;
        case TENSORFUSE_INT32: return 4;
        case TENSORFUSE_FP8_E4M3: return 1;
        case TENSORFUSE_FP8_E5M2: return 1;
        default: return 0;
    }
}

/**
 * @brief Calculate tensor size in bytes
 * @param shape Tensor shape
 * @param dtype Data type
 * @return Size in bytes
 */
static inline size_t tensorfuse_tensor_size_bytes(const TensorFuseShape* shape, TensorFuseDataType dtype) {
    size_t total_elements = 1;
    for (int i = 0; i < shape->ndims; i++) {
        total_elements *= shape->dims[i];
    }
    return total_elements * tensorfuse_sizeof_dtype(dtype);
}

#ifdef __cplusplus
}

// C++ type aliases for convenience
namespace tensorfuse {
    using Status = TensorFuseStatus;
    using DataType = TensorFuseDataType;
    using Layout = TensorFuseLayout;
    using Activation = TensorFuseActivation;
    using Shape = TensorFuseShape;
    using Tensor = TensorFuseTensor;
    using Config = TensorFuseConfig;
    using ModelConfig = TensorFuseModelConfig;
    using TransformerParams = TensorFuseTransformerParams;
    using ProfileConfig = TensorFuseProfileConfig;
    using Metrics = TensorFuseMetrics;
    using KernelConfig = TensorFuseKernelConfig;
    using AutotunerConfig = TensorFuseAutotunerConfig;

    // Device information structure for C++
    struct DeviceInfo {
        int device_id;
        std::string device_name;
        int compute_capability_major;
        int compute_capability_minor;
        size_t total_global_memory;
        int multiprocessor_count;
        int max_threads_per_block;
        size_t max_shared_memory_per_block;
        bool supports_tensor_cores;
        bool supports_fp16;
        bool supports_bf16;
        bool supports_fp8;
    };

    // Memory statistics structure
    struct MemoryStats {
        size_t total_allocated;
        size_t free_memory;
        size_t used_memory;
        size_t peak_usage;
        size_t num_blocks;
    };
}
#endif 