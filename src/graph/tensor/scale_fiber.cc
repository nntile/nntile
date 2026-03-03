/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale_fiber.cc
 * TensorGraph scale_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_fiber.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_scale_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis, Index batch_ndim,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::scale_fiber<T>(alpha, src_t, dst_t, axis, batch_ndim);
}

} // namespace

TensorGraph::TensorNode* scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "scale_fiber: input tensor must be non-null");
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        dst_shape,
        output_name,
        src->dtype());

    scale_fiber(alpha, src, dst, axis, batch_ndim);

    return dst;
}

void scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "scale_fiber: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must have the same dtype");
    }
    // Shape compatibility validated at tensor execution time

    auto op = std::make_shared<TensorScaleFiberOp>(
        alpha, src, dst, axis, batch_ndim);
    src->graph()->add_op(op);
}

void TensorScaleFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale_fiber<nntile::fp32_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::FP64:
            run_scale_fiber<nntile::fp64_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::FP16:
            run_scale_fiber<nntile::fp16_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::BF16:
            run_scale_fiber<nntile::bf16_t>(
                runtime, alpha, axis, batch_ndim, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for scale_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for scale_fiber");
    }
}

} // namespace nntile::graph
