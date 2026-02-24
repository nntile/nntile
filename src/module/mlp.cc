/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/mlp.cc
 * MLP module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/mlp.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

//! Constructor: creates MLP with specified dimensions
Mlp::Mlp(graph::NNGraph& graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         Index output_dim,
         graph::DataType dtype)
    : Module(graph, name)
    , fc1_(graph, name + "_fc1", input_dim, intermediate_dim, dtype)
    , gelu_(graph, name + "_gelu")
    , fc2_(graph, name + "_fc2", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("fc1", &fc1_);
    register_module("gelu", &gelu_);
    register_module("fc2", &fc2_);
}

//! Constructor: creates MLP where output_dim == input_dim
Mlp::Mlp(graph::NNGraph& graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         graph::DataType dtype)
    : Mlp(graph, name, input_dim, intermediate_dim, input_dim, dtype)
{
}

//! Build forward operations
graph::NNGraph::TensorNode& Mlp::build_forward(graph::NNGraph::TensorNode& input)
{
    // Store input reference
    input_tensor_ = &input;

    // fc1: input -> hidden
    hidden_tensor_ = &fc1_.build_forward(input);

    // GELU activation: hidden -> activation
    activation_tensor_ = &gelu_.build_forward(*hidden_tensor_);

    // fc2: activation -> output
    output_tensor_ = &fc2_.build_forward(*activation_tensor_);

    forward_built_ = true;
    return *output_tensor_;
}

//! Build backward operations using gradient tracking
void Mlp::build_backward()
{
    // Check that forward has been built
    if(!forward_built_)
    {
        throw std::runtime_error(
            "Mlp::build_backward: forward not built - "
            "call build_forward first");
    }

    // Backward through fc2
    fc2_.build_backward();

    // Get gradient of activation tensor (output of GELU, input of fc2)
    graph::NNGraph::TensorNode* grad_activation =
        activation_tensor_->grad();
    if(!grad_activation)
    {
        throw std::runtime_error(
            "Mlp::build_backward: no gradient for activation tensor");
    }

    // Backward through GELU
    gelu_.build_backward();

    // Backward through fc1
    fc1_.build_backward();
}

//! Get string representation with dimensions
std::string Mlp::repr() const
{
    return "Mlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) + ")";
}

} // namespace nntile::module
