/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/relu_backward.cc
 * TensorGraph relu_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/relu_backward.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/relu_backward.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_relu_backward(TensorGraph::Runtime& runtime,
                      TensorGraph::TensorNode* x, TensorGraph::TensorNode* dy,
                      TensorGraph::TensorNode* dx)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& dy_t = runtime.get_tensor<T>(dy);
    auto& dx_t = runtime.get_tensor<T>(dx);
    nntile::tensor::relu_backward<T>(x_t, dy_t, dx_t);
}

} // namespace

TensorGraph::TensorNode* relu_backward(TensorGraph::TensorNode* x,
                                       TensorGraph::TensorNode* dy,
                                       const std::string& output_name)
{
    if(x == nullptr || dy == nullptr)
        throw std::invalid_argument("relu_backward: inputs must be non-null");
    if(x->graph() != dy->graph())
        throw std::invalid_argument("relu_backward: inputs must belong to same graph");
    if(x->dtype() != dy->dtype())
        throw std::invalid_argument("relu_backward: inputs must have same dtype");
    if(x->shape() != dy->shape())
        throw std::invalid_argument("relu_backward: x and dy must have same shape");
    if(x == dy)
        throw std::invalid_argument("relu_backward: x and dy must be distinct tensors");
    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape), output_name, x->dtype());
    relu_backward(x, dy, output);
    return output;
}

void relu_backward(TensorGraph::TensorNode* x, TensorGraph::TensorNode* dy,
                   TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
        throw std::invalid_argument("relu_backward: tensors must be non-null");
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
        throw std::invalid_argument("relu_backward: tensors must belong to same graph");
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
        throw std::invalid_argument("relu_backward: tensors must have same dtype");
    if(x->shape() != dy->shape() || x->shape() != dx->shape())
        throw std::invalid_argument("relu_backward: x, dy, dx must have same shape");
    if(x == dy || x == dx || dy == dx)
        throw std::invalid_argument("relu_backward: x, dy, and dx must be distinct tensors");
    auto op = std::make_shared<TensorReluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorReluBackwardOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);
    switch(dtype)
    {
        case DataType::FP32: run_relu_backward<nntile::fp32_t>(runtime, x, dy, dx); break;
        case DataType::FP32_FAST_TF32: run_relu_backward<nntile::fp32_fast_tf32_t>(runtime, x, dy, dx); break;
        case DataType::FP32_FAST_FP16: run_relu_backward<nntile::fp32_fast_fp16_t>(runtime, x, dy, dx); break;
        case DataType::FP32_FAST_BF16: run_relu_backward<nntile::fp32_fast_bf16_t>(runtime, x, dy, dx); break;
        case DataType::FP64: run_relu_backward<nntile::fp64_t>(runtime, x, dy, dx); break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 not supported for relu_backward (use FP32_FAST_FP16)");
            break;
        case DataType::BF16: run_relu_backward<nntile::bf16_t>(runtime, x, dy, dx); break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(std::string(dtype_to_string(dtype)) +
                " not supported for relu_backward");
        default: throw std::runtime_error("Unsupported data type for relu_backward");
    }
}

} // namespace nntile::graph::tensor
