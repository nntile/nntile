/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu_inplace.cc
 * TensorGraph gelu_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu_inplace.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_gelu_inplace(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* dst)
{
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::gelu_inplace<T>(dst_t);
}

} // namespace

void gelu_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "gelu_inplace: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorGeluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TensorGeluInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu_inplace<nntile::fp32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu_inplace<nntile::fp32_fast_tf32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu_inplace<nntile::fp32_fast_fp16_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu_inplace<nntile::fp32_fast_bf16_t>(runtime, dst);
            break;
        case DataType::FP64:
            run_gelu_inplace<nntile::fp64_t>(runtime, dst);
            break;
        case DataType::FP16:
            run_gelu_inplace<nntile::fp16_t>(runtime, dst);
            break;
        case DataType::BF16:
            run_gelu_inplace<nntile::bf16_t>(runtime, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu_inplace");
    }
}

} // namespace nntile::graph
