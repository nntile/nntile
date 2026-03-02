/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph/add_fiber.cc
 * NNGraph add_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/add_fiber.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_fiber.hh"
#include "nntile/graph/tensor/add_fiber_inplace.hh"
#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/sum_fiber.hh"

namespace nntile::graph
{

void NNAddFiberOp::add_forward_to_tensor_graph(NNGraph& graph)
{
    (void)graph;
    if(fiber == nullptr || tensor == nullptr || output == nullptr)
    {
        throw std::invalid_argument(
            "NNAddFiberOp::add_forward_to_tensor_graph: fiber, tensor, output "
            "must be non-null");
    }
    auto tg_op = std::make_shared<TensorAddFiberOp>(
        fiber->data(), tensor->data(), output->data(),
        alpha, beta, axis, batch_ndim);
    fiber->graph().tensor_graph().add_op(tg_op);
}

void NNAddFiberOp::backward()
{
    NNGraph& graph = fiber->graph();
    NNGraph::TensorNode* grad_out = output->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(fiber != nullptr && fiber->requires_grad())
    {
        bool first = graph.is_first_grad(fiber);
        NNGraph::TensorNode* grad_fiber =
            graph.get_or_create_grad(fiber, fiber->name() + "_grad");
        graph::sum_fiber(grad_out->data(), grad_fiber->data(), axis,
                        batch_ndim, 0, alpha, first ? 0.0 : 1.0);
    }
    if(tensor != nullptr && tensor->requires_grad())
    {
        NNGraph::TensorNode* grad_tensor =
            graph.get_or_create_grad(tensor, tensor->name() + "_grad");
        graph::add_inplace(beta, grad_out->data(), Scalar(1.0),
                          grad_tensor->data());
    }
}

NNGraph::TensorNode* add_fiber(
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
            "add_fiber: fiber and tensor must be non-null");
    }
    NNGraph& graph = fiber->graph();
    std::vector<Index> output_shape = tensor->shape();
    bool out_requires_grad = any_input_requires_grad({fiber, tensor});
    NNGraph::TensorNode* output = graph.tensor(
        std::move(output_shape), output_name, tensor->dtype(), out_requires_grad);

    auto op = std::make_shared<NNAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    op->add_forward_to_tensor_graph(graph);
    register_op(graph, std::move(op));
    return output;
}

} // namespace nntile::graph
