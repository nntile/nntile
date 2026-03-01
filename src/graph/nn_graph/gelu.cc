/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph/gelu.cc
 * NNGraph GELU autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/gelu.hh"
#include "nntile/graph/logical/clear.hh"
#include "nntile/graph/logical/gelu.hh"
#include "nntile/graph/logical/gelu_backward.hh"

#include <stdexcept>

namespace nntile::graph
{

ForwardResult Gelu::build_forward(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("Gelu::build_forward: x must be non-null");
    }
    LogicalGraph::TensorNode& y_data = gelu(x->data(), output_name);
    return {{&y_data}, {x}, GeluAttrs{}};
}

void Gelu::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& inputs = op->inputs();
    if(inputs.size() >= 1 && grad_out != nullptr)
    {
        NNGraph::TensorNode* x_nn = inputs[0];
        if(x_nn != nullptr && x_nn->requires_grad())
        {
            bool first = graph.is_first_grad(x_nn);
            NNGraph::TensorNode* grad_x =
                graph.get_or_create_grad(x_nn, x_nn->name() + "_grad");
            if(first)
            {
                clear(grad_x->data());
            }
            gelu_backward(x_nn->data(), grad_out->data(), grad_x->data());
        }
    }
}

} // namespace nntile::graph
