/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_mlp.hh
 * LlamaMLP module implementation using NNTile graph API.
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
#include <nntile/graph/nn/multiply.hh>
#include <nntile/graph/nn/silu.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::llama
{

//! LlamaMLP (SwiGLU) module using graph API
//!
//! Architecture: gate_proj -> SiLU, up_proj, then multiply -> down_proj
//!   - gate_proj: input_dim -> intermediate_dim
//!   - up_proj: input_dim -> intermediate_dim
//!   - gate_act = SiLU(gate_proj(input))
//!   - hidden = gate_act * up_proj(input)
//!   - down_proj: intermediate_dim -> output_dim
//!
//! This matches the Llama MLP from Hugging Face transformers.
class LlamaMLP : public module::ModuleBase
{
private:
    //! Gate projection: input -> intermediate
    module::Linear gate_proj_;

    //! Up projection: input -> intermediate
    module::Linear up_proj_;

    //! Down projection: intermediate -> output
    module::Linear down_proj_;

    //! Dimensions
    Index input_dim_;
    Index intermediate_dim_;
    Index output_dim_;
    graph::DataType dtype_;

    //! Intermediate tensors (created during build_forward)
    graph::NNGraph::TensorNode* gate_tensor_ = nullptr;
    graph::NNGraph::TensorNode* gate_act_tensor_ = nullptr;
    graph::NNGraph::TensorNode* up_tensor_ = nullptr;
    graph::NNGraph::TensorNode* hidden_tensor_ = nullptr;

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor: creates LlamaMLP with specified dimensions
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension (gate/up projections)
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for all tensors
    LlamaMLP(graph::NNGraph& graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
             Index output_dim,
             graph::DataType dtype = graph::DataType::FP32);

    //! Constructor: creates LlamaMLP where output_dim == input_dim (common in transformers)
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param dtype Data type for all tensors
    LlamaMLP(graph::NNGraph& graph,
             const std::string& name,
             Index input_dim,
             Index intermediate_dim,
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
    module::Linear& gate_proj() { return gate_proj_; }
    const module::Linear& gate_proj() const { return gate_proj_; }
    module::Linear& up_proj() { return up_proj_; }
    const module::Linear& up_proj() const { return up_proj_; }
    module::Linear& down_proj() { return down_proj_; }
    const module::Linear& down_proj() const { return down_proj_; }
};

} // namespace nntile::model::llama
