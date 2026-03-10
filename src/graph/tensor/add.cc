/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add.cc
 * TensorGraph add operation implementation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/tensor/add.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor.hh>
#include <nntile/tensor/add.hh>

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add(
    TensorGraph::Runtime& runtime,
    Scalar alpha,
    Scalar beta,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    auto& z_t = runtime.get_tensor<T>(z);
    nntile::tensor::add<T>(alpha, x_t, beta, y_t, z_t);
}

} // namespace

TensorGraph::TensorNode* add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("add: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "add: input tensors must belong to the same graph");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "add: x and y must be distinct tensors");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same dtype");
    }
    if(x->ndim() != y->ndim())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same ndim");
    }

    for(Index i = 0; i < x->ndim(); ++i)
    {
        merge_axis(x->mutable_axes()[i], y->mutable_axes()[i]);
    }

    TensorGraph::TensorNode* output = x->graph()->data(
        x->shape(), output_name, x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorAddOp>(x, y, output, alpha, beta);
    x->graph()->add_op(op);

    return output;
}

void add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z)
{
    if(x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument("add: input tensors must be non-null");
    }
    if(x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "add: x, y, and z must be distinct tensors");
    }
    if(x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "add: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype() || x->dtype() != z->dtype())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same dtype");
    }
    if(x->ndim() != y->ndim() || x->ndim() != z->ndim())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same ndim");
    }

    for(Index i = 0; i < x->ndim(); ++i)
    {
        merge_axis(x->mutable_axes()[i], y->mutable_axes()[i]);
        merge_axis(x->mutable_axes()[i], z->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorAddOp>(x, y, z, alpha, beta);
    x->graph()->add_op(op);
}

void TensorAddOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = x->dtype();

    switch(dtype)
    {
        case DataType::FP32:
            run_add<nntile::fp32_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_add<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_add<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_add<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP64:
            run_add<nntile::fp64_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP16:
            run_add<nntile::fp16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::BF16:
            run_add<nntile::bf16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add operation");
        default:
            throw std::runtime_error("Unsupported data type for add");
    }
}

} // namespace nntile::graph::tensor
