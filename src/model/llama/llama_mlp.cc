/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_mlp.cc
 * LlamaMLP module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/model/llama/llama_mlp.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::model::llama
{

//! Constructor: creates LlamaMLP with specified dimensions
LlamaMLP::LlamaMLP(graph::NNGraph& graph,
                   const std::string& name,
                   Index input_dim,
                   Index intermediate_dim,
                   Index output_dim,
                   graph::DataType dtype)
    : module::ModuleBase(graph, name)
    , gate_proj_(graph, name + "_gate_proj", input_dim, intermediate_dim, dtype)
    , up_proj_(graph, name + "_up_proj", input_dim, intermediate_dim, dtype)
    , down_proj_(graph, name + "_down_proj", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("down_proj", &down_proj_);
}

//! Constructor: creates LlamaMLP where output_dim == input_dim
LlamaMLP::LlamaMLP(graph::NNGraph& graph,
                   const std::string& name,
                   Index input_dim,
                   Index intermediate_dim,
                   graph::DataType dtype)
    : LlamaMLP(graph, name, input_dim, intermediate_dim, input_dim, dtype)
{
}

graph::NNGraph::TensorNode& LlamaMLP::build_forward(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;
    gate_tensor_ = &gate_proj_(input);
    gate_act_tensor_ = graph::silu(gate_tensor_, tensor_name("gate_act"));
    up_tensor_ = &up_proj_(input);
    hidden_tensor_ = graph::multiply(
        gate_act_tensor_, up_tensor_, tensor_name("hidden"));
    output_tensor_ = &down_proj_(*hidden_tensor_);
    return *output_tensor_;
}

//! Get string representation with dimensions
std::string LlamaMLP::repr() const
{
    return "LlamaMLP(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) + ")";
}

} // namespace nntile::model::llama
