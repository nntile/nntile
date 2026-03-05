/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/norm_fiber.cc
 * NNGraph norm_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/norm_fiber.hh"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/norm_fiber.hh"

namespace nntile::graph
{

namespace
{

std::vector<Index> norm_fiber_output_shape(
    const std::vector<Index>& x_shape,
    Index axis,
    Index batch_ndim)
{
    Index ndim = static_cast<Index>(x_shape.size());
    std::vector<Index> out_shape;
    out_shape.reserve(batch_ndim + 1);
    out_shape.push_back(x_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[ndim - batch_ndim + i]);
    }
    return out_shape;
}

} // anonymous namespace

NNGraph::TensorNode* NNNormFiberOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNNormFiberOp::forward: x must be non-null");
    }
    Index ndim = static_cast<Index>(x->shape().size());
    if(axis < 0 || axis >= ndim)
    {
        throw std::invalid_argument(
            "NNNormFiberOp::forward: axis must be in [0, ndim), got axis="
            + std::to_string(axis) + ", ndim=" + std::to_string(ndim));
    }
    if(batch_ndim < 0 || batch_ndim > ndim)
    {
        throw std::invalid_argument(
            "NNNormFiberOp::forward: batch_ndim must be in [0, ndim], got "
            "batch_ndim=" + std::to_string(batch_ndim)
            + ", ndim=" + std::to_string(ndim));
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    std::vector<Index> base_shape =
        norm_fiber_output_shape(x->shape(), axis, batch_ndim);
    NNGraph::TensorNode* base = graph->tensor(
        std::move(base_shape), output_name + "_base", x->dtype(), false);
    graph::tensor::clear(base->data());
    constexpr Scalar beta_fresh = 0.0;  // NNGraph always outputs fresh data
    TensorGraph::TensorNode* y_data = graph::tensor::norm_fiber(
        alpha, x->data(), beta_fresh, base->data(),
        output_name, axis, batch_ndim, redux);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNNormFiberOp::backward() const
{
    if(x != nullptr && x->requires_grad())
    {
        throw std::runtime_error(
            "norm_fiber backward is not implemented");
    }
}

NNGraph::TensorNode* norm_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("norm_fiber: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNNormFiberOp>(
        x, axis, batch_ndim, redux, alpha);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
