/*
 * Copyright (c) 2024, TensorFuse Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tensor operations Python bindings for TensorFuse
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "tensorfuse/tensorfuse.h"
#include "tensorfuse/types.h"

namespace py = pybind11;

void bind_tensor(py::module& m) {
    
    // ==============================================================================
    // Tensor Class
    // ==============================================================================
    
    py::class_<TensorFuseTensor>(m, "Tensor")
        .def(py::init<>())
        .def_readwrite("shape", &TensorFuseTensor::shape)
        .def_readwrite("dtype", &TensorFuseTensor::dtype)
        .def_readwrite("layout", &TensorFuseTensor::layout)
        .def_readwrite("device_id", &TensorFuseTensor::device_id)
        .def_readwrite("size_bytes", &TensorFuseTensor::size_bytes)
        .def_readwrite("is_contiguous", &TensorFuseTensor::is_contiguous)
        .def_property("data", 
            [](const TensorFuseTensor& self) {
                // Return data as a pointer (for advanced users)
                return reinterpret_cast<uintptr_t>(self.data);
            },
            [](TensorFuseTensor& self, uintptr_t ptr) {
                self.data = reinterpret_cast<void*>(ptr);
            })
        .def("__repr__", [](const TensorFuseTensor& self) {
            std::string result = "Tensor(shape=";
            result += "[";
            for (int i = 0; i < self.shape.ndims; ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(self.shape.dims[i]);
            }
            result += "], dtype=";
            result += std::to_string(static_cast<int>(self.dtype));
            result += ", device_id=";
            result += std::to_string(self.device_id);
            result += ")";
            return result;
        });
    
    // ==============================================================================
    // Tensor Creation Functions
    // ==============================================================================
    
    m.def("create_tensor", [](const std::vector<int>& shape, TensorFuseDataType dtype, int device_id) {
        TensorFuseShape tf_shape = {};
        tf_shape.ndims = std::min(static_cast<int>(shape.size()), TENSORFUSE_MAX_DIMS);
        std::copy(shape.begin(), shape.begin() + tf_shape.ndims, tf_shape.dims);
        
        TensorFuseTensor tensor = {};
        TensorFuseStatus status = tensorfuse_allocate_tensor(&tensor, &tf_shape, dtype, device_id);
        if (status != TENSORFUSE_SUCCESS) {
            throw std::runtime_error("Failed to create tensor: " + 
                                     std::string(tensorfuse_get_error_string(status)));
        }
        return tensor;
    }, "Create tensor with specified shape and dtype", 
       py::arg("shape"), py::arg("dtype"), py::arg("device_id") = 0);
    
    m.def("from_numpy", [](const py::array& arr, int device_id) {
        TensorFuseTensor tensor = {};
        
        // Validate input array
        if (arr.size() == 0) {
            throw std::runtime_error("Cannot create tensor from empty array");
        }
        
        if (arr.ndim() == 0) {
            throw std::runtime_error("Cannot create tensor from scalar array");
        }
        
        if (arr.ndim() > TENSORFUSE_MAX_DIMS) {
            throw std::runtime_error("Array has too many dimensions (max: " + 
                                     std::to_string(TENSORFUSE_MAX_DIMS) + ", got: " + 
                                     std::to_string(arr.ndim()) + ")");
        }
        
        // Set shape
        tensor.shape.ndims = arr.ndim();
        for (int i = 0; i < arr.ndim() && i < TENSORFUSE_MAX_DIMS; ++i) {
            tensor.shape.dims[i] = arr.shape(i);
        }
        
        // Set data type
        py::dtype dtype = arr.dtype();
        if (dtype.is(py::dtype::of<float>())) {
            tensor.dtype = TENSORFUSE_FLOAT32;
        } else if (dtype.is(py::dtype("float16"))) {
            tensor.dtype = TENSORFUSE_FLOAT16;
        } else if (dtype.is(py::dtype("int8"))) {
            tensor.dtype = TENSORFUSE_INT8;
        } else if (dtype.is(py::dtype("uint8"))) {
            tensor.dtype = TENSORFUSE_UINT8;
        } else if (dtype.is(py::dtype("int32"))) {
            tensor.dtype = TENSORFUSE_INT32;
        } else {
            throw std::runtime_error("Unsupported numpy dtype");
        }
        
        // Set other properties
        tensor.layout = TENSORFUSE_LAYOUT_ROW_MAJOR;
        tensor.device_id = device_id;
        tensor.size_bytes = arr.size() * arr.itemsize();
        tensor.is_contiguous = true;
        tensor.scale = nullptr;
        tensor.zero_point = nullptr;
        
        // Copy data to GPU (placeholder - in real implementation would use CUDA)
        // For now, just reference the numpy array data
        tensor.data = const_cast<void*>(arr.data());
        
        return tensor;
    }, "Create tensor from numpy array", py::arg("array"), py::arg("device_id") = 0);
    
    m.def("to_numpy", [](const TensorFuseTensor& tensor) {
        // Create numpy array from tensor (placeholder implementation)
        std::vector<py::ssize_t> shape(tensor.shape.dims, tensor.shape.dims + tensor.shape.ndims);
        
        py::dtype dtype;
        switch (tensor.dtype) {
            case TENSORFUSE_FLOAT32:
                dtype = py::dtype::of<float>();
                break;
            case TENSORFUSE_FLOAT16:
                dtype = py::dtype("float16");
                break;
            case TENSORFUSE_INT8:
                dtype = py::dtype::of<int8_t>();
                break;
            case TENSORFUSE_UINT8:
                dtype = py::dtype::of<uint8_t>();
                break;
            case TENSORFUSE_INT32:
                dtype = py::dtype::of<int32_t>();
                break;
            default:
                throw std::runtime_error("Unsupported tensor dtype");
        }
        
        return py::array(dtype, shape, tensor.data);
    }, "Convert tensor to numpy array", py::arg("tensor"));
    
    // ==============================================================================
    // Tensor Utilities
    // ==============================================================================
    
    m.def("tensor_copy", [](const TensorFuseTensor& src, TensorFuseTensor& dst) {
        // Placeholder for tensor copy implementation
        // In real implementation, this would use cudaMemcpy
        if (src.size_bytes != dst.size_bytes) {
            throw std::runtime_error("Tensor size mismatch");
        }
        std::memcpy(dst.data, src.data, src.size_bytes);
    }, "Copy tensor data", py::arg("src"), py::arg("dst"));
    
    m.def("tensor_fill", [](TensorFuseTensor& tensor, float value) {
        // Placeholder for tensor fill implementation
        // In real implementation, this would launch a CUDA kernel
        if (tensor.dtype != TENSORFUSE_FLOAT32) {
            throw std::runtime_error("Only float32 tensors supported for fill");
        }
        float* data = static_cast<float*>(tensor.data);
        size_t num_elements = tensor.size_bytes / sizeof(float);
        std::fill(data, data + num_elements, value);
    }, "Fill tensor with value", py::arg("tensor"), py::arg("value"));
    
    m.def("tensor_zero", [](TensorFuseTensor& tensor) {
        // Placeholder for tensor zero implementation
        // In real implementation, this would use cudaMemset
        std::memset(tensor.data, 0, tensor.size_bytes);
    }, "Fill tensor with zeros", py::arg("tensor"));
    
    m.def("tensor_reshape", [](TensorFuseTensor& tensor, const std::vector<int>& new_shape) {
        // Check if reshape is valid
        size_t old_size = 1;
        for (int i = 0; i < tensor.shape.ndims; ++i) {
            old_size *= tensor.shape.dims[i];
        }
        
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        
        if (old_size != new_size) {
            throw std::runtime_error("Invalid reshape: size mismatch");
        }
        
        // Update shape
        tensor.shape.ndims = std::min(static_cast<int>(new_shape.size()), TENSORFUSE_MAX_DIMS);
        std::copy(new_shape.begin(), new_shape.begin() + tensor.shape.ndims, tensor.shape.dims);
    }, "Reshape tensor", py::arg("tensor"), py::arg("new_shape"));
} 