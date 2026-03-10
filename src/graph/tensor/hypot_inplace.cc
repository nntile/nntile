/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot_inplace.cc
 * TensorGraph hypot_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_hypot_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::hypot_inplace<T>(alpha, src_t, beta, dst_t);
}

} // namespace

void hypot_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "hypot_inplace: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must have the same dtype");
    }
    if(src->ndim() != dst->ndim())
    {
        throw std::invalid_argument(
            "hypot_inplace: src and dst must have the same ndim");
    }

    for(Index i = 0; i < src->ndim(); ++i)
    {
        merge_axis(src->mutable_axes()[i], dst->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorHypotInplaceOp>(
        alpha, beta, src, dst);
    src->graph()->add_op(op);
}

void TensorHypotInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot_inplace<nntile::fp32_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot_inplace<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot_inplace<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot_inplace<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP64:
            run_hypot_inplace<nntile::fp64_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for hypot_inplace operation");
        case DataType::BF16:
            run_hypot_inplace<nntile::bf16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot_inplace");
    }
}

} // namespace nntile::graph::tensor
