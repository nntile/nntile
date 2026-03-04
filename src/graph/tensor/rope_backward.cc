/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/rope_backward.cc
 * TensorGraph rope_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/rope_backward.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/rope_backward.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_rope_backward(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    auto& sin_t = runtime.get_tensor<T>(sin);
    auto& cos_t = runtime.get_tensor<T>(cos);
    auto& dy_t = runtime.get_tensor<T>(dy);
    auto& dx_t = runtime.get_tensor<T>(dx);
    nntile::tensor::rope_backward<T>(sin_t, cos_t, dy_t, dx_t);
}

} // namespace

TensorGraph::TensorNode* rope_backward(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* dy,
    const std::string& output_name)
{
    if(sin == nullptr || cos == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must be non-null");
    }
    if(sin->graph() != cos->graph() || sin->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must belong to the same graph");
    }
    if(sin->dtype() != cos->dtype() || sin->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must have the same dtype");
    }

    std::vector<Index> output_shape = dy->shape();
    TensorGraph::TensorNode* dx = dy->graph()->data(
        std::move(output_shape),
        output_name,
        dy->dtype());

    rope_backward(sin, cos, dy, dx);

    return dx;
}

void rope_backward(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    if(sin == nullptr || cos == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must be non-null");
    }
    if(sin->graph() != cos->graph() || sin->graph() != dy->graph() ||
       sin->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must belong to the same graph");
    }
    if(sin->dtype() != cos->dtype() || sin->dtype() != dy->dtype() ||
       sin->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must have the same dtype");
    }
    if(dy->shape() != dx->shape())
    {
        throw std::invalid_argument(
            "rope_backward: dx must have the same shape as dy");
    }

    auto op = std::make_shared<TensorRopeBackwardOp>(sin, cos, dy, dx);
    dy->graph()->add_op(op);
}

void TensorRopeBackwardOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dy);

    switch(dtype)
    {
        case DataType::FP32:
            run_rope_backward<nntile::fp32_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run_rope_backward<nntile::fp32_fast_tf32_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run_rope_backward<nntile::fp32_fast_fp16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run_rope_backward<nntile::fp32_fast_bf16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP64:
            run_rope_backward<nntile::fp64_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP16:
            run_rope_backward<nntile::fp16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::BF16:
            run_rope_backward<nntile::bf16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for rope_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for rope_backward");
    }
}

} // namespace nntile::graph::tensor
