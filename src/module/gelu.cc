/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/gelu.cc
 * GeLU module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/gelu.hh"
#include "nntile/graph/logical/clear.hh"
#include "nntile/graph/logical/gelu_backward.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

Gelu::Gelu(graph::NNGraph& graph, const std::string& name)
    : Module(graph, name)
{
}

graph::NNGraph::TensorNode& Gelu::build_forward(
    graph::NNGraph::TensorNode& input)
{
    return forward(input);
}

graph::NNGraph::TensorNode& Gelu::forward_impl(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;
    output_tensor_ = graph::gelu(&input, tensor_name("output"));
    return *output_tensor_;
}

std::vector<graph::NNGraph::TensorNode*> Gelu::backward_inputs() const
{
    return {input_tensor_};
}

void Gelu::build_backward(const graph::NNGraph::OpNode* op)
{
    graph::NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& inputs = op->inputs();
    if(inputs.empty() || grad_out == nullptr)
    {
        return;
    }
    graph::NNGraph::TensorNode* x_nn = inputs[0];
    if(x_nn != nullptr && graph_.requires_grad(x_nn))
    {
        bool first = graph_.is_first_grad(x_nn);
        graph::NNGraph::TensorNode* grad_x =
            graph_.get_or_create_grad(x_nn, x_nn->name() + "_grad");
        if(first)
        {
            graph::clear(grad_x->data());
        }
        graph::gelu_backward(x_nn->data(), grad_out->data(), grad_x->data());
    }
}

std::string Gelu::repr() const
{
    return "Gelu()";
}

} // namespace nntile::module
