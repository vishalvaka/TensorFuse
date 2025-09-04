/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Memory management Python bindings for TensorFuse
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "tensorfuse/tensorfuse.h"
#include "tensorfuse/types.h"

namespace py = pybind11;

void bind_memory(py::module& m) {
    
    // ==============================================================================
    // Memory Management
    // ==============================================================================
    
    m.def("allocate_tensor", [](const TensorFuseShape& shape, TensorFuseDataType dtype, int device_id) {
        TensorFuseTensor tensor = {};
        TensorFuseStatus status = tensorfuse_allocate_tensor(&tensor, &shape, dtype, device_id);
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to allocate tensor: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
        return tensor;
    }, "Allocate tensor with optimal memory alignment", 
       py::arg("shape"), py::arg("dtype"), py::arg("device_id") = 0);
    
    m.def("free_tensor", [](TensorFuseTensor& tensor) {
        TensorFuseStatus status = tensorfuse_free_tensor(&tensor);
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to free tensor: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
    }, "Free tensor memory", py::arg("tensor"));
    
    // ==============================================================================
    // Memory Utilities
    // ==============================================================================
    
    m.def("get_memory_info", []() {
        py::dict info;
        
        // Get GPU memory info (placeholder implementation)
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        
        // In real implementation, this would call cudaMemGetInfo
        // cudaMemGetInfo(&free_bytes, &total_bytes);
        
        info["free_bytes"] = free_bytes;
        info["total_bytes"] = total_bytes;
        info["used_bytes"] = total_bytes - free_bytes;
        
        return info;
    }, "Get GPU memory information");
    
    m.def("get_tensor_size_bytes", [](const TensorFuseShape& shape, TensorFuseDataType dtype) {
        return tensorfuse_tensor_size_bytes(&shape, dtype);
    }, "Calculate tensor size in bytes", py::arg("shape"), py::arg("dtype"));
    
    m.def("get_dtype_size", [](TensorFuseDataType dtype) {
        return tensorfuse_sizeof_dtype(dtype);
    }, "Get size of data type in bytes", py::arg("dtype"));
    
    // ==============================================================================
    // Memory Pool Management
    // ==============================================================================
    
    m.def("set_memory_pool_size", [](size_t size_bytes) {
        // Placeholder for setting memory pool size
        // In real implementation, this would configure the memory pool
    }, "Set memory pool size", py::arg("size_bytes"));
    
    m.def("get_memory_pool_stats", []() {
        py::dict stats;
        stats["pool_size_bytes"] = 0;
        stats["allocated_bytes"] = 0;
        stats["free_bytes"] = 0;
        stats["num_allocations"] = 0;
        stats["num_frees"] = 0;
        return stats;
    }, "Get memory pool statistics");
    
    m.def("clear_memory_pool", []() {
        // Placeholder for clearing memory pool
        // In real implementation, this would clear the memory pool
    }, "Clear memory pool");
} 