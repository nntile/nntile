/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/dtype.cc
 * Implementation of DataType utilities.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/dtype.hh"

#include <stdexcept>

namespace nntile::graph
{

//! Convert DataType to string
std::string dtype_to_string(DataType dtype)
{
    switch(dtype)
    {
        case DataType::FP32:
            return "FP32";
        case DataType::FP32_FAST_TF32:
            return "FP32_FAST_TF32";
        case DataType::FP32_FAST_FP16:
            return "FP32_FAST_FP16";
        case DataType::FP32_FAST_BF16:
            return "FP32_FAST_BF16";
        case DataType::FP64:
            return "FP64";
        case DataType::FP16:
            return "FP16";
        case DataType::BF16:
            return "BF16";
        case DataType::INT64:
            return "INT64";
        case DataType::INT32:
            return "INT32";
        case DataType::BOOL:
            return "BOOL";
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype)
{
    switch(dtype)
    {
        // 1 byte
        case DataType::BOOL:
            return 1;
        // 2 bytes
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        // 4 bytes
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::INT32:
            return 4;
        // 8 bytes
        case DataType::FP64:
        case DataType::INT64:
            return 8;
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

} // namespace nntile::graph
