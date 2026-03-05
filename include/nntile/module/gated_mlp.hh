/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/gated_mlp.hh
 * Gated MLP module (Llama-style) using NNTile graph API.
 *
 * Architecture: down_proj(activation(gate_proj(x)) * up_proj(x))
 *   - gate_proj: input_dim -> intermediate_dim
 *   - up_proj: input_dim -> intermediate_dim
 *   - activation: silu (default, Llama-style)
 *   - element-wise multiply: gate * up
 *   - down_proj: intermediate_dim -> output_dim
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/torch.h>
#endif

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Gated MLP (Llama-style) module using graph API
//!
//! Architecture: down_proj(activation(gate_proj(x)) * up_proj(x))
//! Uses silu activation by default (Llama/Mistral style).
class GatedMlp : public Module
{
private:
    //! Gate projection: input -> intermediate
    Linear gate_proj_;

    //! Up projection: input -> intermediate
    Linear up_proj_;

    //! Activation (default: silu for Llama-style)
    Activation activation_;

    //! Down projection: intermediate -> output
    Linear down_proj_;

    //! Dimensions
    Index input_dim_;
    Index intermediate_dim_;
    Index output_dim_;
    graph::DataType dtype_;

    //! Intermediate tensors (created during forward)
    graph::NNGraph::TensorNode* gate_tensor_ = nullptr;   // After gate_proj
    graph::NNGraph::TensorNode* up_tensor_ = nullptr;    // After up_proj
    graph::NNGraph::TensorNode* gate_act_tensor_ = nullptr;  // After activation
    graph::NNGraph::TensorNode* hidden_tensor_ = nullptr;    // gate_act * up

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor: creates GatedMLP with specified dimensions
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param output_dim Output feature dimension
    //! @param activation Activation for gate (default: silu, Llama-style)
    //! @param dtype Data type for all tensors
    GatedMlp(graph::NNGraph* graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
             Index output_dim,
             ActivationType activation = ActivationType::SILU,
             graph::DataType dtype = graph::DataType::FP32);

    //! Constructor: creates GatedMLP where output_dim == input_dim (common in transformers)
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param activation Activation for gate (default: silu, Llama-style)
    //! @param dtype Data type for all tensors
    GatedMlp(graph::NNGraph* graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
             ActivationType activation = ActivationType::SILU,
             graph::DataType dtype = graph::DataType::FP32);

#ifdef NNTILE_HAVE_TORCH
    //! Constructor: creates GatedMlp from PyTorch Linear layers with
    //! automatic weight/bias binding for easy data transfer.
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param gate_proj_layer PyTorch gate projection (input -> intermediate)
    //! @param up_proj_layer PyTorch up projection (input -> intermediate)
    //! @param down_proj_layer PyTorch down projection (intermediate -> output)
    //! @param activation Activation for gate (must match PyTorch)
    //! @param dtype Data type for all tensors
    GatedMlp(graph::NNGraph* graph,
             const std::string& name,
             const torch::nn::Linear& gate_proj_layer,
             const torch::nn::Linear& up_proj_layer,
             const torch::nn::Linear& down_proj_layer,
             ActivationType activation = ActivationType::SILU,
             graph::DataType dtype = graph::DataType::FP32);
#endif

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input);

    //! Forward: calls forward
    graph::NNGraph::TensorNode* operator()(graph::NNGraph::TensorNode* input)
    {
        return forward(input);
    }

    //! Get string representation with dimensions
    std::string repr() const override;

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index intermediate_dim() const { return intermediate_dim_; }
    Index output_dim() const { return output_dim_; }

    // Submodule accessors
    Linear& gate_proj() { return gate_proj_; }
    const Linear& gate_proj() const { return gate_proj_; }
    Linear& up_proj() { return up_proj_; }
    const Linear& up_proj() const { return up_proj_; }
    Activation& activation() { return activation_; }
    const Activation& activation() const { return activation_; }
    Linear& down_proj() { return down_proj_; }
    const Linear& down_proj() const { return down_proj_; }
};

#ifdef NNTILE_HAVE_TORCH

inline GatedMlp::GatedMlp(graph::NNGraph* graph,
                         const std::string& name,
                         const torch::nn::Linear& gate_proj_layer,
                         const torch::nn::Linear& up_proj_layer,
                         const torch::nn::Linear& down_proj_layer,
                         ActivationType activation,
                         graph::DataType dtype)
    : Module(graph, name)
    , gate_proj_(graph, name + "_gate_proj", gate_proj_layer, dtype)
    , up_proj_(graph, name + "_up_proj", up_proj_layer, dtype)
    , activation_(graph, name + "_activation", activation)
    , down_proj_(graph, name + "_down_proj", down_proj_layer, dtype)
    , input_dim_(static_cast<Index>(gate_proj_layer->weight.size(1)))
    , intermediate_dim_(static_cast<Index>(gate_proj_layer->weight.size(0)))
    , output_dim_(static_cast<Index>(down_proj_layer->weight.size(0)))
    , dtype_(dtype)
{
    if(static_cast<Index>(up_proj_layer->weight.size(1)) != input_dim_ ||
       static_cast<Index>(up_proj_layer->weight.size(0)) != intermediate_dim_)
    {
        throw std::invalid_argument(
            "GatedMlp::GatedMlp: up_proj dimensions must match gate_proj");
    }
    if(static_cast<Index>(down_proj_layer->weight.size(1)) != intermediate_dim_)
    {
        throw std::invalid_argument(
            "GatedMlp::GatedMlp: down_proj input dim must match intermediate");
    }
    register_module("gate_proj", &gate_proj_);
    register_module("up_proj", &up_proj_);
    register_module("activation", &activation_);
    register_module("down_proj", &down_proj_);
}

#endif // NNTILE_HAVE_TORCH

} // namespace nntile::module
