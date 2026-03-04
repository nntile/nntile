/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelutanh_backward.cc
 * TensorGraph gelutanh_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelutanh_backward.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelutanh_backward.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_gelutanh_backward(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& dy_t = runtime.get_tensor<T>(dy);
    auto& dx_t = runtime.get_tensor<T>(dx);
    nntile::tensor::gelutanh_backward<T>(x_t, dy_t, dx_t);
}

} // namespace

TensorGraph::TensorNode* gelutanh_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    const std::string& output_name)
{
    if(x == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must have the same dtype");
    }
    if(x->shape() != dy->shape())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must have the same shape");
    }
    if(x == dy)
    {
        throw std::invalid_argument(
            "gelutanh_backward: x and dy must be distinct tensors");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* dx = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    gelutanh_backward(x, dy, dx);

    return dx;
}

void gelutanh_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must have the same dtype");
    }
    if(x->shape() != dx->shape())
    {
        throw std::invalid_argument(
            "gelutanh_backward: dx must have the same shape as x");
    }
    if(x == dy || x == dx || dy == dx)
    {
        throw std::invalid_argument(
            "gelutanh_backward: x, dy, and dx must be distinct tensors");
    }

    auto op = std::make_shared<TensorGelutanhBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorGelutanhBackwardOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh_backward<nntile::fp32_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh_backward<nntile::fp32_fast_tf32_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh_backward<nntile::fp32_fast_fp16_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh_backward<nntile::fp32_fast_bf16_t>(runtime, x, dy, dx);
            break;
        case DataType::FP64:
            run_gelutanh_backward<nntile::fp64_t>(runtime, x, dy, dx);
            break;
        case DataType::FP16:
            run_gelutanh_backward<nntile::fp16_t>(runtime, x, dy, dx);
            break;
        case DataType::BF16:
            run_gelutanh_backward<nntile::bf16_t>(runtime, x, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh_backward operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for gelutanh_backward");
    }
}

} // namespace nntile::graph::tensor
