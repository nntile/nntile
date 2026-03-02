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
#include <nntile/graph/execution_context.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/tensor/add.hh>

namespace nntile::graph
{

namespace
{

template<typename T>
void run_add(
    ExecutionContext<TensorGraph::DataNode>& ctx,
    Scalar alpha,
    Scalar beta,
    TensorGraph::DataNode* x,
    TensorGraph::DataNode* y,
    TensorGraph::DataNode* z)
{
    auto& x_t = ctx.get_tensor<T>(x);
    auto& y_t = ctx.get_tensor<T>(y);
    auto& z_t = ctx.get_tensor<T>(z);
    nntile::tensor::add<T>(alpha, x_t, beta, y_t, z_t);
}

} // namespace

TensorGraph::DataNode* add(
    Scalar alpha,
    TensorGraph::DataNode* x,
    Scalar beta,
    TensorGraph::DataNode* y,
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
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same dtype");
    }
    if(x->shape() != y->shape())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same shape");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::DataNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    auto op = std::make_shared<TensorAddOp>(x, y, output, alpha, beta);

    x->graph()->add_op(op);

    return output;
}

void TensorAddOp::execute(
    ExecutionContext<TensorGraph::DataNode>& ctx) const
{
    DataType dtype = ctx.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_add<nntile::fp32_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_add<nntile::fp32_fast_tf32_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_add<nntile::fp32_fast_fp16_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_add<nntile::fp32_fast_bf16_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::FP64:
            run_add<nntile::fp64_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::FP16:
            run_add<nntile::fp16_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::BF16:
            run_add<nntile::bf16_t>(ctx, alpha, beta, x, y, z);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add operation");
        default:
            throw std::runtime_error("Unsupported data type for add");
    }
}

} // namespace nntile::graph
