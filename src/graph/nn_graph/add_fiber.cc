/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph/add_fiber.cc
 * NNGraph add_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/add_fiber.hh"
#include "nntile/graph/logical/add_fiber_inplace.hh"
#include "nntile/graph/logical/add_inplace.hh"
#include "nntile/graph/logical/sum_fiber.hh"

#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode* AddFiber::build_forward(
    Scalar alpha,
    NNGraph::TensorNode* fiber,
    Scalar beta,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "AddFiber::build_forward: fiber and tensor must be non-null");
    }
    NNGraph& graph = fiber->graph();
    LogicalGraph::TensorNode& out_data = add_fiber(
        alpha, fiber->data(), beta, tensor->data(), output_name, axis,
        batch_ndim);
    bool out_requires_grad = any_input_requires_grad({fiber, tensor});
    NNGraph::TensorNode* out = graph.tensor(out_data, out_requires_grad);
    register_op(graph, {fiber, tensor}, out,
                std::make_shared<AddFiberAttrs>(AddFiberAttrs{axis, batch_ndim, alpha, beta}),
                [](const NNGraph::OpNode* op) { AddFiber::build_backward(op); },
                {});
    return out;
}

void AddFiber::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& attrs = *std::static_pointer_cast<AddFiberAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    Scalar beta = attrs.beta;
    Index axis = attrs.axis;
    Index batch_ndim = attrs.batch_ndim;
    const auto& inputs = op->inputs();
    if(inputs.size() < 2 || grad_out == nullptr)
    {
        return;
    }
    NNGraph::TensorNode* fiber_nn = inputs[0];
    NNGraph::TensorNode* tensor_nn = inputs[1];

    // grad_fiber += alpha * sum_fiber(grad_out)
    if(fiber_nn != nullptr && fiber_nn->requires_grad())
    {
        bool first = graph.is_first_grad(fiber_nn);
        NNGraph::TensorNode* grad_fiber =
            graph.get_or_create_grad(fiber_nn, fiber_nn->name() + "_grad");
        sum_fiber(grad_out->data(), grad_fiber->data(), axis, batch_ndim, 0,
                  alpha, first ? 0.0 : 1.0);
    }

    // grad_tensor += beta * grad_out
    if(tensor_nn != nullptr && tensor_nn->requires_grad())
    {
        NNGraph::TensorNode* grad_tensor =
            graph.get_or_create_grad(tensor_nn, tensor_nn->name() + "_grad");
        add_inplace(beta, grad_out->data(), Scalar(1.0), grad_tensor->data());
    }
}

} // namespace nntile::graph
