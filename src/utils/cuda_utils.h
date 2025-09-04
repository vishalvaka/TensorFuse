/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA utilities and helper functions
 */

#pragma once

#include "tensorfuse/types.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace tensorfuse {
namespace utils {

/**
 * @brief Get the number of CUDA devices
 * @return Number of available CUDA devices
 */
int get_device_count();

/**
 * @brief Get device properties for a specific device
 * @param device_id Device ID
 * @return Device properties
 */
cudaDeviceProp get_device_properties(int device_id);

/**
 * @brief Get device information for a specific device
 * @param device_id Device ID
 * @return DeviceInfo structure
 */
DeviceInfo get_device_info(int device_id);

/**
 * @brief Get information for all available devices
 * @return Vector of DeviceInfo structures
 */
std::vector<DeviceInfo> get_all_device_info();

/**
 * @brief Check if a device supports TensorFuse requirements
 * @param device_id Device ID
 * @return true if device is supported
 */
bool is_device_supported(int device_id);

/**
 * @brief Get the best device for TensorFuse operations
 * @return Device ID of the best device, or -1 if none found
 */
int get_best_device();

/**
 * @brief Get compute capability as a single number (major * 10 + minor)
 * @param device_id Device ID
 * @return Compute capability as integer
 */
int get_compute_capability(int device_id);

/**
 * @brief Check if device supports specific data type
 * @param device_id Device ID
 * @param data_type Data type to check
 * @return true if supported
 */
bool supports_data_type(int device_id, DataType data_type);

/**
 * @brief Get supported data types for a device
 * @param device_id Device ID
 * @return Vector of supported data types
 */
std::vector<DataType> get_supported_data_types(int device_id);

/**
 * @brief Get memory information for a device
 * @param device_id Device ID
 * @param free_memory Output: free memory in bytes
 * @param total_memory Output: total memory in bytes
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus get_memory_info(int device_id, size_t& free_memory, size_t& total_memory);

/**
 * @brief Set the current CUDA device
 * @param device_id Device ID
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus set_device(int device_id);

/**
 * @brief Get the current CUDA device
 * @return Current device ID, or -1 on error
 */
int get_current_device();

/**
 * @brief Synchronize a specific device
 * @param device_id Device ID (-1 for current device)
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus synchronize_device(int device_id = -1);

/**
 * @brief Check if CUDA is available
 * @return true if CUDA is available
 */
bool is_cuda_available();

/**
 * @brief Get CUDA runtime version
 * @return CUDA runtime version
 */
int get_cuda_runtime_version();

/**
 * @brief Get CUDA driver version
 * @return CUDA driver version
 */
int get_cuda_driver_version();

/**
 * @brief Format device name for display
 * @param device_id Device ID
 * @return Formatted device name string
 */
std::string format_device_name(int device_id);

/**
 * @brief Format memory size for display
 * @param bytes Memory size in bytes
 * @return Formatted memory string (e.g., "8.0 GB")
 */
std::string format_memory_size(size_t bytes);

/**
 * @brief Get architecture name from compute capability
 * @param major Compute capability major version
 * @param minor Compute capability minor version
 * @return Architecture name (e.g., "Ampere", "Ada Lovelace", "Hopper")
 */
std::string get_architecture_name(int major, int minor);

/**
 * @brief Check if two devices can access each other's memory
 * @param device1 First device ID
 * @param device2 Second device ID
 * @return true if peer access is possible
 */
bool can_access_peer(int device1, int device2);

/**
 * @brief Enable peer access between two devices
 * @param device1 First device ID
 * @param device2 Second device ID
 * @return TensorFuseStatus indicating success or failure
 */
TensorFuseStatus enable_peer_access(int device1, int device2);

/**
 * @brief Get optimal block size for a kernel
 * @param func CUDA kernel function pointer
 * @param dynamic_shared_mem Dynamic shared memory size
 * @return Optimal block size
 */
int get_optimal_block_size(const void* func, size_t dynamic_shared_mem = 0);

/**
 * @brief Get maximum threads per block for current device
 * @return Maximum threads per block
 */
int get_max_threads_per_block();

/**
 * @brief Get maximum shared memory per block for current device
 * @return Maximum shared memory in bytes
 */
size_t get_max_shared_memory_per_block();

/**
 * @brief Get number of multiprocessors for current device
 * @return Number of multiprocessors
 */
int get_multiprocessor_count();

/**
 * @brief Get warp size for current device
 * @return Warp size (typically 32)
 */
int get_warp_size();

} // namespace utils
} // namespace tensorfuse 