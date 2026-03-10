/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_slice_inplace.cc
 * TensorGraph norm_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_slice_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_norm_slice_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, int redux,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::norm_slice_inplace<T>(
        alpha, src_t, beta, dst_t, axis, redux);
}

} // namespace

void norm_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src, axis, "norm_slice_inplace");

    auto op = std::make_shared<TensorNormSliceInplaceOp>(
        alpha, beta, src, dst, axis, redux);
    src->graph()->add_op(op);
}

void TensorNormSliceInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_slice_inplace<nntile::fp32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_slice_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_slice_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_slice_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP64:
            run_norm_slice_inplace<nntile::fp64_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_slice_inplace operation");
        case DataType::BF16:
            run_norm_slice_inplace<nntile::bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for norm_slice_inplace");
    }
}

} // namespace nntile::graph::tensor
