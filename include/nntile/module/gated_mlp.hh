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
#include <string>
#include <vector>

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

    //! Intermediate tensors (created during build_forward)
    graph::NNGraph::TensorNode* gate_tensor_ = nullptr;   // After gate_proj
    graph::NNGraph::TensorNode* up_tensor_ = nullptr;    // After up_proj
    graph::NNGraph::TensorNode* gate_act_tensor_ = nullptr;  // After activation
    graph::NNGraph::TensorNode* hidden_tensor_ = nullptr;    // gate_act * up

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor: creates GatedMLP with specified dimensions
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param output_dim Output feature dimension
    //! @param activation Activation for gate (default: silu, Llama-style)
    //! @param dtype Data type for all tensors
    GatedMlp(graph::NNGraph& graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
             Index output_dim,
             ActivationType activation = ActivationType::SILU,
             graph::DataType dtype = graph::DataType::FP32);

    //! Constructor: creates GatedMLP where output_dim == input_dim (common in transformers)
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param activation Activation for gate (default: silu, Llama-style)
    //! @param dtype Data type for all tensors
    GatedMlp(graph::NNGraph& graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
             ActivationType activation = ActivationType::SILU,
             graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    //! Forward: calls build_forward
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& input)
    {
        return build_forward(input);
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

} // namespace nntile::module
