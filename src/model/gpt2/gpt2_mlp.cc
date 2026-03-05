/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/gpt2/gpt2_mlp.cc
 * GPT-2 MLP module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/model/gpt2/gpt2_mlp.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::model::gpt2
{

//! Constructor: creates GPT-2 MLP with specified dimensions
Gpt2Mlp::Gpt2Mlp(graph::NNGraph& graph,
                 const std::string& name,
                 Index input_dim,
                 Index intermediate_dim,
                 Index output_dim,
                 graph::DataType dtype)
    : module::ModuleBase(graph, name)
    , c_fc_(graph, name + "_c_fc", input_dim, intermediate_dim, dtype)
    , gelu_(graph, name + "_gelu")
    , c_proj_(graph, name + "_c_proj", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("c_fc", &c_fc_);
    register_module("gelu", &gelu_);
    register_module("c_proj", &c_proj_);
}

//! Constructor: creates GPT-2 MLP where output_dim == input_dim
Gpt2Mlp::Gpt2Mlp(graph::NNGraph& graph,
                 const std::string& name,
                 Index input_dim,
                 Index intermediate_dim,
                 graph::DataType dtype)
    : Gpt2Mlp(graph, name, input_dim, intermediate_dim, input_dim, dtype)
{
}

graph::NNGraph::TensorNode& Gpt2Mlp::build_forward(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;
    hidden_tensor_ = &c_fc_(input);
    activation_tensor_ = &gelu_(*hidden_tensor_);
    output_tensor_ = &c_proj_(*activation_tensor_);
    return *output_tensor_;
}

//! Get string representation with dimensions
std::string Gpt2Mlp::repr() const
{
    return "Gpt2Mlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) + ")";
}

} // namespace nntile::model::gpt2
