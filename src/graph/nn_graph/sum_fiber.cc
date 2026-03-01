/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph/sum_fiber.cc
 * NNGraph sum_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/sum_fiber.hh"
#include "nntile/graph/logical/add_fiber_inplace.hh"
#include "nntile/graph/logical/clear.hh"
#include "nntile/graph/logical/sum_fiber.hh"

#include <stdexcept>
#include <vector>

namespace nntile::graph
{

namespace
{

//! Output shape for sum_fiber: [shape[axis], shape[ndim-batch_ndim], ...]
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

NNGraph::TensorNode* SumFiber::build_forward(
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
        throw std::invalid_argument("SumFiber::build_forward: x must be non-null");
    }
    NNGraph& graph = x->graph();
    LogicalGraph& logical = graph.logical_graph();
    std::vector<Index> y_shape =
        sum_fiber_output_shape(x->shape(), axis, batch_ndim);
    LogicalGraph::TensorNode& y_data =
        logical.tensor(y_shape, output_name, x->dtype());
    clear(y_data);
    sum_fiber(x->data(), y_data, axis, batch_ndim, redux, alpha, beta);
    bool out_requires_grad = any_input_requires_grad({x});
    NNGraph::TensorNode* y = graph.tensor(y_data, out_requires_grad);
    register_op(graph, {x}, y,
                std::make_shared<ReductionAttrs>(ReductionAttrs{alpha, beta, axis, batch_ndim, redux}),
                [](const NNGraph::OpNode* op) { SumFiber::build_backward(op); },
                {});
    return y;
}

void SumFiber::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& attrs = *std::static_pointer_cast<ReductionAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    Index axis = attrs.axis;
    Index batch_ndim = attrs.batch_ndim;
    const auto& inputs = op->inputs();
    if(inputs.size() >= 1 && grad_out != nullptr)
    {
        NNGraph::TensorNode* x_nn = inputs[0];
        if(x_nn != nullptr && x_nn->requires_grad())
        {
            NNGraph::TensorNode* grad_x =
                graph.get_or_create_grad(x_nn, x_nn->name() + "_grad");
            add_fiber_inplace(alpha, grad_out->data(), Scalar(1.0),
                             grad_x->data(), axis, batch_ndim);
        }
    }
}

} // namespace nntile::graph
