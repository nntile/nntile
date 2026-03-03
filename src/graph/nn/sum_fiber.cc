/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/sum_fiber.cc
 * NNGraph sum_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/sum_fiber.hh"

#include <stdexcept>
#include <vector>

#include "nntile/graph/tensor/add_fiber_inplace.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/sum_fiber.hh"

namespace nntile::graph
{

namespace
{

std::vector<Index> sum_fiber_output_shape(const std::vector<Index>& x_shape,
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

NNGraph::TensorNode* NNSumFiberOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNSumFiberOp::forward: x must be non-null");
    }
    NNGraph& graph = x->graph();
    std::vector<Index> y_shape =
        sum_fiber_output_shape(x->shape(), axis, batch_ndim);
    bool out_requires_grad = any_input_requires_grad({x});
    NNGraph::TensorNode* y = graph.tensor(
        std::move(y_shape), output_name, x->dtype(), out_requires_grad);
    outputs_ = {y};
    graph::clear(y->data());
    graph::sum_fiber(x->data(), y->data(), axis, batch_ndim, redux, alpha, beta);
    return y;
}

void NNSumFiberOp::backward() const
{
    NNGraph& graph = x->graph();
    NNGraph::TensorNode* grad_out = output()->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x != nullptr && x->requires_grad())
    {
        NNGraph::TensorNode* grad_x =
            graph.get_or_create_grad(x, x->name() + "_grad");
        graph::add_fiber_inplace(alpha, grad_out->data(), Scalar(1.0),
                                grad_x->data(), axis, batch_ndim);
    }
}

NNGraph::TensorNode* sum_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("sum_fiber: x must be non-null");
    }
    NNGraph& graph = x->graph();
    auto op = std::make_shared<NNSumFiberOp>(
        x, axis, batch_ndim, redux, alpha, beta);
    NNGraph::TensorNode* y = op->forward(output_name);
    register_op(graph, std::move(op));
    return y;
}

} // namespace nntile::graph
