/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor_spec.cc
 * Implementation of TensorSpec class.
 *
 * @version 1.1.0
 * */

#include <nntile/graph/tensor_spec.hh>
#include <stdexcept>
#include <numeric>

namespace nntile::graph {

//! Convert DataType to string
std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        case DataType::FP16: return "FP16";
        case DataType::BF16: return "BF16";
        case DataType::INT64: return "INT64";
        case DataType::INT32: return "INT32";
        case DataType::BOOL: return "BOOL";
        default: throw std::invalid_argument("Unknown DataType");
    }
}

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP64: return 8;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT64: return 8;
        case DataType::INT32: return 4;
        case DataType::BOOL: return 1;
        default: throw std::invalid_argument("Unknown DataType");
    }
}

//! Tensor specification - shape and data type
TensorSpec::TensorSpec(std::vector<Index> shape, DataType dtype)
    : shape_(std::move(shape))
    , dtype_(dtype)
{
    // Validate shape
    for (Index dim : shape_) {
        if (dim <= 0) {
            throw std::invalid_argument("TensorSpec: all dimensions must be positive");
        }
    }
}

//! Get dimension at index (supports negative indexing)
Index TensorSpec::dim(int idx) const {
    if (idx < 0) {
        idx += static_cast<int>(shape_.size());
    }
    if (idx < 0 || static_cast<size_t>(idx) >= shape_.size()) {
        throw std::out_of_range("TensorSpec::dim: index out of range");
    }
    return shape_[static_cast<size_t>(idx)];
}

//! Total number of elements
Index TensorSpec::nelems() const {
    return std::accumulate(shape_.begin(), shape_.end(), Index(1),
                          std::multiplies<Index>());
}

//! Total size in bytes
size_t TensorSpec::size_bytes() const {
    return static_cast<size_t>(nelems()) * dtype_size(dtype_);
}

//! Check if shapes are compatible for operations
bool TensorSpec::is_compatible(const TensorSpec& other) const {
    return dtype_ == other.dtype_;
}

//! String representation
std::string TensorSpec::to_string() const {
    std::string result = "TensorSpec([";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape_[i]);
    }
    result += "], " + dtype_to_string(dtype_) + ")";
    return result;
}

} // namespace nntile::graph
