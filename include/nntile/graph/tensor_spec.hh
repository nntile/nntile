/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor_spec.hh
 * TensorSpec class for logical graph tensor specifications.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include third-party headers

// Include other NNTile headers
#include <nntile/base_types.hh>

namespace nntile::graph
{

//! Data types supported
enum class DataType {
    FP32,
    FP32_FAST_TF32,
    FP32_FAST_FP16,
    FP32_FAST_BF16,
    FP64,
    FP16,
    BF16,
    INT64,
    INT32,
    BOOL
};

//! Convert DataType to string
std::string dtype_to_string(DataType dtype);

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype);

//! Tensor specification - shape and data type
class TensorSpec
{
private:
    std::vector<Index> shape_;
    DataType dtype_;

public:
    //! Construct with shape and dtype
    TensorSpec(std::vector<Index> shape, DataType dtype = DataType::FP32);

    //! Accessors
    const std::vector<Index>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    Index ndim() const { return static_cast<Index>(shape_.size()); }

    //! Get dimension at index (supports negative indexing)
    Index dim(int idx) const;

    //! Total number of elements
    Index nelems() const;

    //! Total size in bytes
    size_t size_bytes() const;

    //! Check if shapes are compatible for operations
    bool is_compatible(const TensorSpec& other) const;

    //! String representation
    std::string to_string() const;
};

} // namespace nntile::graph
