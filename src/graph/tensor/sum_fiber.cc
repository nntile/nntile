/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum_fiber.cc
 * TensorGraph sum_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum_fiber.hh"

#include <stdexcept>
#include <utility>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_fiber.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_sum_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim, int redux,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    nntile::tensor::sum_fiber<T>(
        alpha, x_t, beta, y_t, axis, batch_ndim, redux);
}

std::vector<Index> sum_fiber_output_shape(
    const std::vector<Index>& x_shape,
    Index axis,
    Index batch_ndim)
{
    Index ndim = x_shape.size();
    std::vector<Index> out_shape;
    out_shape.reserve(batch_ndim + 1);
    out_shape.push_back(x_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[ndim - batch_ndim + i]);
    }
    return out_shape;
}

} // namespace

TensorGraph::TensorNode* sum_fiber(
    TensorGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "sum_fiber: input tensor must be non-null");
    }

    std::vector<Index> output_shape =
        sum_fiber_output_shape(x->shape(), axis, batch_ndim);
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    // Merge output fiber axes with x axes
    merge_axis(output->mutable_axes()[0],
               x->mutable_axes()[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        merge_axis(output->mutable_axes()[1 + i],
                   x->mutable_axes()[x->ndim() - batch_ndim + i]);
    }

    auto op = std::make_shared<TensorSumFiberOp>(
        x, output, axis, batch_ndim, redux, alpha, beta);
    x->graph()->add_op(op);

    return output;
}

void sum_fiber(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "sum_fiber: x and y must be distinct tensors");
    }
    validate_fiber_shape_and_merge(y, x, axis, batch_ndim, "sum_fiber");

    auto op = std::make_shared<TensorSumFiberOp>(
        x, y, axis, batch_ndim, redux, alpha, beta);

    x->graph()->add_op(op);
}

void TensorSumFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_fiber<nntile::fp32_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP64:
            run_sum_fiber<nntile::fp64_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP16:
            run_sum_fiber<nntile::fp16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::BF16:
            run_sum_fiber<nntile::bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_fiber");
    }
}

} // namespace nntile::graph::tensor
