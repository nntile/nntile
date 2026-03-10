/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_slice_inplace.cc
 * TensorGraph add_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_slice_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_slice_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::add_slice_inplace<T>(alpha, src_t, beta, dst_t, axis);
}

} // namespace

void add_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must have the same dtype");
    }
    if(axis < 0 || axis >= dst->ndim())
    {
        throw std::invalid_argument(
            "add_slice_inplace: axis out of range");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "add_slice_inplace: src and dst must be distinct tensors");
    }
    if(src->ndim() + 1 != dst->ndim())
    {
        throw std::invalid_argument(
            "add_slice_inplace: src must have ndim = dst.ndim - 1");
    }

    // Merge slice broadcast: src with dst
    int d = 0;
    for(Index i = 0; i < dst->ndim(); ++i)
    {
        if(i == axis) continue;
        merge_axis(src->mutable_axes()[static_cast<size_t>(d)],
                   dst->mutable_axes()[static_cast<size_t>(i)]);
        ++d;
    }

    auto op = std::make_shared<TensorAddSliceInplaceOp>(
        src, dst, alpha, beta, axis);
    src->graph()->add_op(op);
}

void TensorAddSliceInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_slice_inplace<nntile::fp32_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_slice_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_slice_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_slice_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP64:
            run_add_slice_inplace<nntile::fp64_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP16:
            run_add_slice_inplace<nntile::fp16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::BF16:
            run_add_slice_inplace<nntile::bf16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_slice_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for add_slice_inplace");
    }
}

} // namespace nntile::graph::tensor
