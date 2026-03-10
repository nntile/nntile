/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_fiber.cc
 * TensorGraph add_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_fiber.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim,
    TensorGraph::TensorNode* fiber,
    TensorGraph::TensorNode* tensor,
    TensorGraph::TensorNode* output)
{
    auto& fiber_t = runtime.get_tensor<T>(fiber);
    auto& tensor_t = runtime.get_tensor<T>(tensor);
    auto& output_t = runtime.get_tensor<T>(output);
    nntile::tensor::add_fiber<T>(
        alpha, fiber_t, beta, tensor_t, output_t, axis, batch_ndim);
}

} // namespace

TensorGraph::TensorNode* add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }

    validate_fiber_broadcast_shape_and_merge(fiber, tensor, axis, batch_ndim, "add_fiber");

    // Output shape matches tensor (fiber is broadcast)
    std::vector<Index> output_shape = tensor->shape();
    TensorGraph::TensorNode* output = tensor->graph()->data(
        std::move(output_shape),
        output_name,
        tensor->dtype());
    output->set_axes(tensor->axes());

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);

    return output;
}

void add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    TensorGraph::TensorNode* output,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr || output == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph() || fiber->graph() != output->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype() || fiber->dtype() != output->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }
    if(fiber == tensor || fiber == output || tensor == output)
    {
        throw std::invalid_argument(
            "add_fiber: fiber, tensor, and output must be distinct tensors");
    }
    validate_fiber_broadcast_shape_and_merge(fiber, tensor, axis, batch_ndim, "add_fiber");
    validate_same_shape_and_merge(tensor, output, "add_fiber");

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);
}

void TensorAddFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(fiber);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_fiber<nntile::fp32_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        case DataType::FP64:
            run_add_fiber<nntile::fp64_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_fiber operation");
        case DataType::BF16:
            run_add_fiber<nntile::bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor, output);
            break;
        default:
            throw std::runtime_error("Unsupported data type for add_fiber");
    }
}

} // namespace nntile::graph::tensor
