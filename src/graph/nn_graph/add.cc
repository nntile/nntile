/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph/add.cc
 * NNGraph add operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/add.hh"
#include "nntile/graph/nn_graph/tensor_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add.hh"
#include "nntile/graph/tensor/add_inplace.hh"

namespace nntile::graph
{

void NNAddOp::add_forward_to_tensor_graph(NNGraph& graph)
{
    (void)graph;
    if(x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument(
            "NNAddOp::add_forward_to_tensor_graph: x, y, z must be non-null");
    }
    auto tg_op = std::make_shared<TensorAddOp>(
        x->data(), y->data(), z->data(), alpha, beta);
    x->graph().tensor_graph().add_op(tg_op);
}

void NNAddOp::backward()
{
    NNGraph& graph = x->graph();
    NNGraph::TensorNode* grad_out = z->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x != nullptr && x->requires_grad())
    {
        NNGraph::TensorNode* grad_x =
            graph.get_or_create_grad(x, x->name() + "_grad");
        graph::add_inplace(alpha, grad_out->data(), Scalar(1.0), grad_x->data());
    }
    if(y != nullptr && y->requires_grad())
    {
        NNGraph::TensorNode* grad_y =
            graph.get_or_create_grad(y, y->name() + "_grad");
        graph::add_inplace(beta, grad_out->data(), Scalar(1.0), grad_y->data());
    }
}

NNGraph::TensorNode* add(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("add: x and y must be non-null");
    }
    NNGraph& graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph.tensor(
        x->shape(), output_name, x->dtype(), out_requires_grad);

    auto op = std::make_shared<NNAddOp>(x, y, z, alpha, beta);
    op->add_forward_to_tensor_graph(graph);
    register_op(graph, std::move(op));
    return z;
}

} // namespace nntile::graph
