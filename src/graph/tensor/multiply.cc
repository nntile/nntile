/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply.cc
 * TensorGraph multiply operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/execution_context.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_multiply(
    ExecutionContext<TensorGraph::TensorNode>& ctx,
    Scalar alpha,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z)
{
    auto& x_t = ctx.get_tensor<T>(x);
    auto& y_t = ctx.get_tensor<T>(y);
    auto& z_t = ctx.get_tensor<T>(z);
    nntile::tensor::multiply<T>(alpha, x_t, y_t, z_t);
}

} // namespace

TensorGraph::TensorNode* multiply(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "multiply: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "multiply: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "multiply: input tensors must have the same dtype");
    }
    if(x->shape() != y->shape())
    {
        throw std::invalid_argument(
            "multiply: input tensors must have the same shape");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    auto op = std::make_shared<TensorMultiplyOp>(x, y, output, 1.0);
    x->graph()->add_op(op);

    return output;
}

void TensorMultiplyOp::execute(
    ExecutionContext<TensorGraph::TensorNode>& ctx) const
{
    DataType dtype = ctx.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply<nntile::fp32_t>(ctx, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply<nntile::fp32_fast_tf32_t>(ctx, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply<nntile::fp32_fast_fp16_t>(ctx, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply<nntile::fp32_fast_bf16_t>(ctx, alpha, x, y, z);
            break;
        case DataType::FP64:
            run_multiply<nntile::fp64_t>(ctx, alpha, x, y, z);
            break;
        case DataType::FP16:
            run_multiply<nntile::fp16_t>(ctx, alpha, x, y, z);
            break;
        case DataType::BF16:
            run_multiply<nntile::bf16_t>(ctx, alpha, x, y, z);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                "INT64/BOOL data type not supported for multiply operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply");
    }
}

} // namespace nntile::graph
