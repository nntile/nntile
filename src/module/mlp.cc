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
Mlp::Mlp(graph::NNGraph* graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         Index output_dim,
         ActivationType activation,
         graph::DataType dtype)
    : Module(graph, name)
    , fc1_(graph, name + "_fc1", input_dim, intermediate_dim, dtype)
    , activation_(graph, name + "_activation", activation)
    , fc2_(graph, name + "_fc2", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("fc1", &fc1_);
    register_module("activation", &activation_);
    register_module("fc2", &fc2_);
}

//! Constructor: creates MLP where output_dim == input_dim
Mlp::Mlp(graph::NNGraph* graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         ActivationType activation,
         graph::DataType dtype)
    : Mlp(graph, name, input_dim, intermediate_dim, input_dim, activation, dtype)
{
}

graph::NNGraph::TensorNode* Mlp::forward(
    graph::NNGraph::TensorNode* input)
{
    if(input == nullptr)
    {
        throw std::invalid_argument(
            "Mlp::forward: input tensor must be non-null");
    }
    input_tensor_ = input;
    hidden_tensor_ = fc1_(input);
    activation_tensor_ = activation_(hidden_tensor_);
    output_tensor_ = fc2_(activation_tensor_);
    return output_tensor_;
}

//! Get string representation with dimensions
std::string Mlp::repr() const
{
    return "Mlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) +
           ", activation=" + activation_type_to_string(activation_.type()) + ")";
}

} // namespace nntile::module
