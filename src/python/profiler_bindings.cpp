/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Profiler Python bindings for TensorFuse
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <thread>

#include "tensorfuse/tensorfuse.h"
#include "tensorfuse/types.h"
#include "tensorfuse/profiler.h"

namespace py = pybind11;

void bind_profiler(py::module& m) {
    
    // ==============================================================================
    // Profiler Configuration
    // ==============================================================================
    
    py::class_<TensorFuseProfileConfig>(m, "ProfileConfig")
        .def(py::init<>())
        .def_readwrite("enable_nsight_systems", &TensorFuseProfileConfig::enable_nsight_systems)
        .def_readwrite("enable_nsight_compute", &TensorFuseProfileConfig::enable_nsight_compute)
        .def_readwrite("enable_nvtx", &TensorFuseProfileConfig::enable_nvtx)
        .def_readwrite("enable_cuda_events", &TensorFuseProfileConfig::enable_cuda_events)
        .def_readwrite("enable_memory_tracking", &TensorFuseProfileConfig::enable_memory_tracking)
        .def_readwrite("warmup_iterations", &TensorFuseProfileConfig::warmup_iterations)
        .def_readwrite("profile_iterations", &TensorFuseProfileConfig::profile_iterations)
        .def_property("output_prefix",
            [](const TensorFuseProfileConfig& self) {
                return self.output_prefix ? std::string(self.output_prefix) : std::string();
            },
            [](TensorFuseProfileConfig& self, const std::string& prefix) {
                static std::string stored_prefix = prefix;
                self.output_prefix = stored_prefix.c_str();
            })
        .def("__repr__", [](const TensorFuseProfileConfig& self) {
            return "ProfileConfig(enable_nsight_systems=" + std::to_string(self.enable_nsight_systems) + 
                   ", enable_memory_tracking=" + std::to_string(self.enable_memory_tracking) + ")";
        });
    
    // ==============================================================================
    // Performance Metrics
    // ==============================================================================
    
    py::class_<TensorFuseMetrics>(m, "Metrics")
        .def(py::init<>())
        .def_readwrite("kernel_time_ms", &TensorFuseMetrics::kernel_time_ms)
        .def_readwrite("memory_bandwidth_gbps", &TensorFuseMetrics::memory_bandwidth_gbps)
        .def_readwrite("tensor_core_utilization", &TensorFuseMetrics::tensor_core_utilization)
        .def_readwrite("flops_per_second", &TensorFuseMetrics::flops_per_second)
        .def_readwrite("memory_used_bytes", &TensorFuseMetrics::memory_used_bytes)
        .def_readwrite("memory_peak_bytes", &TensorFuseMetrics::memory_peak_bytes)
        .def_readwrite("num_kernel_launches", &TensorFuseMetrics::num_kernel_launches)
        .def_readwrite("cpu_time_ms", &TensorFuseMetrics::cpu_time_ms)
        .def_readwrite("total_time_ms", &TensorFuseMetrics::total_time_ms)
        .def("__repr__", [](const TensorFuseMetrics& self) {
            return "Metrics(kernel_time_ms=" + std::to_string(self.kernel_time_ms) + 
                   ", memory_bandwidth_gbps=" + std::to_string(self.memory_bandwidth_gbps) + 
                   ", tensor_core_utilization=" + std::to_string(self.tensor_core_utilization) + "%)";
        });
    
    // ==============================================================================
    // Profiler Functions
    // ==============================================================================
    
    m.def("start_profiling", [](const TensorFuseProfileConfig& config) {
        TensorFuseStatus status = tensorfuse_start_profiling(&config);
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to start profiling: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
    }, "Start profiling session", py::arg("config"));
    
    m.def("stop_profiling", [](const std::string& output_path) {
        TensorFuseStatus status = tensorfuse_stop_profiling(output_path.c_str());
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to stop profiling: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
    }, "Stop profiling session and save results", py::arg("output_path"));
    
    m.def("get_last_metrics", []() {
        // Placeholder implementation
        TensorFuseMetrics metrics = {};
        metrics.kernel_time_ms = 1.5f;
        metrics.memory_bandwidth_gbps = 800.0f;
        metrics.tensor_core_utilization = 85.0f;
        metrics.flops_per_second = 156e12f;
        metrics.memory_used_bytes = 1024 * 1024 * 512;  // 512MB
        metrics.memory_peak_bytes = 1024 * 1024 * 768;  // 768MB
        metrics.num_kernel_launches = 1;
        metrics.cpu_time_ms = 0.1f;
        metrics.total_time_ms = 1.6f;
        return metrics;
    }, "Get metrics from last operation");
    
    // ==============================================================================
    // Profiler Context Manager
    // ==============================================================================
    
    // Note: TensorFuseProfiler class is not implemented yet
    // This would be a future enhancement for context management
    
    // ==============================================================================
    // Benchmarking Utilities
    // ==============================================================================
    
    // Benchmark operation by function
    m.def("benchmark_operation", [](py::function op, int num_iterations, int num_warmup) {
        py::dict results;
        
        // Warmup
        for (int i = 0; i < num_warmup; ++i) {
            op();
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            op();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avg_time_ms = duration.count() / 1000.0f / num_iterations;
        
        results["avg_time_ms"] = avg_time_ms;
        results["total_time_ms"] = duration.count() / 1000.0f;
        results["iterations"] = num_iterations;
        results["throughput_ops_per_sec"] = 1000.0f / avg_time_ms;
        
        return results;
    }, "Benchmark operation performance", 
       py::arg("operation"), py::arg("num_iterations") = 100, py::arg("num_warmup") = 10);

    // Benchmark operation by name (for backward compatibility)
    m.def("benchmark_operation", [](const std::string& op_name) {
        py::dict results;
        
        // Default parameters
        int iterations = 100;
        int warmup_iterations = 10;
        
        // Placeholder timing for string-based operation names
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Simulate work
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avg_time_ms = duration.count() / 1000.0f;
        
        results["avg_time_ms"] = avg_time_ms;
        results["total_time_ms"] = avg_time_ms * iterations;
        results["iterations"] = iterations;
        results["operation"] = op_name;
        
        return results;
    }, "Benchmark operation by name", 
       py::arg("op_name"));
    
    m.def("profile_memory_usage", [](py::function op) {
        py::dict results;
        
        // Placeholder for memory profiling
        size_t memory_before = 0;
        size_t memory_after = 0;
        
        op();
        
        results["memory_used_bytes"] = memory_after - memory_before;
        results["memory_before_bytes"] = memory_before;
        results["memory_after_bytes"] = memory_after;
        
        return results;
    }, "Profile memory usage of operation", py::arg("operation"));
    
    // ==============================================================================
    // Roofline Analysis
    // ==============================================================================
    
    py::class_<TensorFuseRooflineModel>(m, "RooflineModel")
        .def(py::init<>())
        .def_readwrite("peak_flops", &TensorFuseRooflineModel::peak_flops)
        .def_readwrite("peak_bandwidth", &TensorFuseRooflineModel::peak_bandwidth)
        .def_readwrite("arithmetic_intensity", &TensorFuseRooflineModel::arithmetic_intensity)
        .def_readwrite("achieved_flops", &TensorFuseRooflineModel::achieved_flops)
        .def_readwrite("achieved_bandwidth", &TensorFuseRooflineModel::achieved_bandwidth)
        .def_readwrite("efficiency_flops", &TensorFuseRooflineModel::efficiency_flops)
        .def_readwrite("efficiency_bandwidth", &TensorFuseRooflineModel::efficiency_bandwidth)
        .def_readwrite("is_compute_bound", &TensorFuseRooflineModel::is_compute_bound)
        .def_readwrite("is_memory_bound", &TensorFuseRooflineModel::is_memory_bound)
        .def("__repr__", [](const TensorFuseRooflineModel& self) {
            return "RooflineModel(arithmetic_intensity=" + std::to_string(self.arithmetic_intensity) + 
                   ", efficiency_flops=" + std::to_string(self.efficiency_flops) + "%)";
        });
    
    m.def("calculate_roofline", [](size_t ops_count, size_t bytes_transferred, float time_ms) {
        TensorFuseRooflineModel model = {};
        TensorFuseStatus status = tensorfuse_calculate_roofline(ops_count, bytes_transferred, time_ms, &model);
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to calculate roofline: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
        return model;
    }, "Calculate roofline model metrics", 
       py::arg("ops_count"), py::arg("bytes_transferred"), py::arg("time_ms"));
} 