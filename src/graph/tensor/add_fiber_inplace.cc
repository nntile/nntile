/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_fiber_inplace.cc
 * TensorGraph add_fiber_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/execution_context.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_fiber_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_add_fiber_inplace(
    ExecutionContext<TensorGraph::TensorNode>& ctx,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim,
    TensorGraph::TensorNode* fiber,
    TensorGraph::TensorNode* tensor)
{
    auto& fiber_t = ctx.get_tensor<T>(fiber);
    auto& tensor_t = ctx.get_tensor<T>(tensor);
    nntile::tensor::add_fiber_inplace<T>(
        alpha, fiber_t, beta, tensor_t, axis, batch_ndim);
}

} // namespace

void add_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorAddFiberInplaceOp>(
        fiber, tensor, alpha, beta, axis, batch_ndim);
    tensor->graph()->add_op(op);
}

void TensorAddFiberInplaceOp::execute(
    ExecutionContext<TensorGraph::TensorNode>& ctx) const
{
    DataType dtype = ctx.get_dtype(fiber);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_fiber_inplace<nntile::fp32_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_fiber_inplace<nntile::fp32_fast_tf32_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_fiber_inplace<nntile::fp32_fast_fp16_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_fiber_inplace<nntile::fp32_fast_bf16_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP64:
            run_add_fiber_inplace<nntile::fp64_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_fiber_inplace operation");
        case DataType::BF16:
            run_add_fiber_inplace<nntile::bf16_t>(
                ctx, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for add_fiber_inplace");
    }
}

} // namespace nntile::graph
