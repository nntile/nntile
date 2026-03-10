/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale_slice.cc
 * TensorGraph scale_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_slice.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_scale_slice(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::scale_slice<T>(alpha, src_t, dst_t, axis);
}

} // namespace

TensorGraph::TensorNode* scale_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    Index axis_size)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "scale_slice: input tensor must be non-null");
    }
    if(axis < 0 || axis > static_cast<Index>(src->shape().size()))
    {
        throw std::invalid_argument(
            "scale_slice: axis out of range");
    }

    std::vector<Index> output_shape;
    output_shape.reserve(src->shape().size() + 1);
    for(Index i = 0; i < axis; ++i)
    {
        output_shape.push_back(src->shape()[i]);
    }
    output_shape.push_back(axis_size);
    for(Index i = axis; i < static_cast<Index>(src->shape().size()); ++i)
    {
        output_shape.push_back(src->shape()[i]);
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    scale_slice(alpha, src, dst, axis);

    return dst;
}

void scale_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "scale_slice: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "scale_slice: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "scale_slice: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "scale_slice: input tensors must have the same dtype");
    }
    if(src->ndim() + 1 != dst->ndim())
    {
        throw std::invalid_argument(
            "scale_slice: dst must have ndim = src.ndim + 1");
    }
    if(axis < 0 || axis >= dst->ndim())
    {
        throw std::invalid_argument(
            "scale_slice: axis out of range");
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

    auto op = std::make_shared<TensorScaleSliceOp>(alpha, src, dst, axis);
    src->graph()->add_op(op);
}

void TensorScaleSliceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale_slice<nntile::fp32_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale_slice<nntile::fp32_fast_tf32_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale_slice<nntile::fp32_fast_fp16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale_slice<nntile::fp32_fast_bf16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP64:
            run_scale_slice<nntile::fp64_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP16:
            run_scale_slice<nntile::fp16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::BF16:
            run_scale_slice<nntile::bf16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for scale_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for scale_slice");
    }
}

} // namespace nntile::graph::tensor
