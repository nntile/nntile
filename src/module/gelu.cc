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
    input_tensor_ = &input;

    std::vector<Index> output_shape = input.shape();
    bool output_requires_grad = graph_.requires_grad(input);

    output_tensor_ = &graph_.tensor(
        output_shape,
        tensor_name("output"),
        input.dtype(),
        output_requires_grad);

    graph::gelu(input, *output_tensor_);

    forward_built_ = true;
    return *output_tensor_;
}

void Gelu::build_backward()
{
    if(!forward_built_ || !input_tensor_ || !output_tensor_)
    {
        throw std::runtime_error(
            "Gelu::build_backward: forward not built - call build_forward first");
    }

    graph::NNGraph::TensorNode* grad_output = output_tensor_->grad();
    if(!grad_output)
    {
        throw std::runtime_error(
            "Gelu::build_backward: no gradient registered for output tensor '" +
            output_tensor_->name() + "'");
    }

    if(graph_.requires_grad(*input_tensor_))
    {
        // Use input tensor's own name for its gradient (input is not a parameter)
        graph::NNGraph::TensorNode& grad_input = graph_.get_or_create_grad(
            *input_tensor_, input_tensor_->name() + "_grad");

        graph::gelu_backward(*input_tensor_, *grad_output, grad_input);
    }
}

std::string Gelu::repr() const
{
    return "Gelu()";
}

} // namespace nntile::module
