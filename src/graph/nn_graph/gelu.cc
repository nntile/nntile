/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph/gelu.cc
 * NNGraph GELU autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/gelu.hh"
#include "nntile/graph/nn_graph/tensor_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/gelu.hh"
#include "nntile/graph/tensor/gelu_backward.hh"

namespace nntile::graph
{

void NNGeluOp::add_forward_to_tensor_graph(NNGraph& graph)
{
    (void)graph;
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "NNGeluOp::add_forward_to_tensor_graph: x, y must be non-null");
    }
    graph::gelu(x->data(), y->data());
}

void NNGeluOp::backward()
{
    NNGraph& graph = x->graph();
    NNGraph::TensorNode* grad_out = y->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x != nullptr && x->requires_grad())
    {
        bool first = graph.is_first_grad(x);
        NNGraph::TensorNode* grad_x =
            graph.get_or_create_grad(x, x->name() + "_grad");
        if(first)
        {
            graph::clear(grad_x->data());
        }
        graph::gelu_backward(x->data(), grad_out->data(), grad_x->data());
    }
}

NNGraph::TensorNode* gelu(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("gelu: x must be non-null");
    }
    NNGraph& graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    NNGraph::TensorNode* y = graph.tensor(
        x->shape(), output_name, x->dtype(), out_requires_grad);

    auto op = std::make_shared<NNGeluOp>(x, y);
    op->add_forward_to_tensor_graph(graph);
    register_op(graph, std::move(op));
    return y;
}

} // namespace nntile::graph
