/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/gated_mlp.cc
 * Gated MLP module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/gated_mlp.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

//! Constructor: creates GatedMLP with specified dimensions
GatedMlp::GatedMlp(graph::NNGraph& graph,
                   const std::string& name,
                   Index input_dim,
                   Index intermediate_dim,
                   Index output_dim,
                   ActivationType activation,
                   graph::DataType dtype)
    : Module(graph, name)
    , gate_proj_(graph, name + "_gate_proj", input_dim, intermediate_dim, dtype)
    , up_proj_(graph, name + "_up_proj", input_dim, intermediate_dim, dtype)
    , activation_(graph, name + "_activation", activation)
    , down_proj_(graph, name + "_down_proj", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("activation", &activation_);
    register_module("down_proj", &down_proj_);
}

//! Constructor: creates GatedMLP where output_dim == input_dim
GatedMlp::GatedMlp(graph::NNGraph& graph,
                   const std::string& name,
                   Index input_dim,
                   Index intermediate_dim,
                   ActivationType activation,
                   graph::DataType dtype)
    : GatedMlp(graph, name, input_dim, intermediate_dim, input_dim, activation,
               dtype)
{
}

graph::NNGraph::TensorNode& GatedMlp::build_forward(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;
    gate_tensor_ = &gate_proj_(input);
    up_tensor_ = &up_proj_(input);
    gate_act_tensor_ = &activation_(*gate_tensor_);
    hidden_tensor_ = graph::multiply(gate_act_tensor_, up_tensor_,
                                    tensor_name("hidden"));
    output_tensor_ = &down_proj_(*hidden_tensor_);
    return *output_tensor_;
}

//! Get string representation with dimensions
std::string GatedMlp::repr() const
{
    return "GatedMlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) +
           ", activation=" + activation_type_to_string(activation_.type()) + ")";
}

} // namespace nntile::module
