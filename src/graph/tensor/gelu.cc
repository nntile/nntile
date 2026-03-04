/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu.cc
 * TensorGraph GeLU operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_gelu(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    nntile::tensor::gelu<T>(x_t, y_t);
}

} // namespace

TensorGraph::TensorNode* gelu(
    TensorGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("gelu: input tensor must be non-null");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    auto op = std::make_shared<TensorGeluOp>(x, output);
    x->graph()->add_op(op);

    return output;
}

void gelu(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("gelu: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "gelu: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "gelu: input tensors must have the same dtype");
    }
    if(x->shape() != y->shape())
    {
        throw std::invalid_argument(
            "gelu: output must have the same shape as input");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "gelu: x and y must be distinct tensors");
    }

    auto op = std::make_shared<TensorGeluOp>(x, y);
    x->graph()->add_op(op);
}

void TensorGeluOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu<nntile::fp32_t>(runtime, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu<nntile::fp32_fast_tf32_t>(runtime, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu<nntile::fp32_fast_fp16_t>(runtime, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu<nntile::fp32_fast_bf16_t>(runtime, x, y);
            break;
        case DataType::FP64:
            run_gelu<nntile::fp64_t>(runtime, x, y);
            break;
        case DataType::FP16:
            run_gelu<nntile::fp16_t>(runtime, x, y);
            break;
        case DataType::BF16:
            run_gelu<nntile::bf16_t>(runtime, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu");
    }
}

} // namespace nntile::graph::tensor
