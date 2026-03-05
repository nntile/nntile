/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gpt2/gpt2_mlp.hh
 * GPT-2 MLP module implementation using NNTile graph API.
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
#include <nntile/module/gelu.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::gpt2
{

//! GPT-2 MLP (Multi-Layer Perceptron) module using graph API
//!
//! Architecture (matches GPT-2 from HuggingFace transformers):
//!   - c_fc: input_dim -> intermediate_dim (typically 4 * input_dim)
//!   - activation: GELU
//!   - c_proj: intermediate_dim -> output_dim (typically input_dim)
//!
//! This is a model-specific module for GPT-2 transformer blocks.
class Gpt2Mlp : public module::ModuleBase
{
private:
    //! First linear layer (c_fc): input -> intermediate
    module::Linear c_fc_;

    //! Activation module: hidden -> activation
    module::Gelu gelu_;

    //! Second linear layer (c_proj): activation -> output
    module::Linear c_proj_;

    //! Dimensions
    Index input_dim_;
    Index intermediate_dim_;
    Index output_dim_;
    graph::DataType dtype_;

    //! Intermediate tensors (created during build_forward)
    graph::NNGraph::TensorNode* hidden_tensor_ = nullptr;      // After c_fc
    graph::NNGraph::TensorNode* activation_tensor_ = nullptr;  // After GELU

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor: creates GPT-2 MLP with specified dimensions
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension (after c_fc)
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for all tensors
    Gpt2Mlp(graph::NNGraph& graph,
            const std::string& name,
            Index input_dim,
            Index intermediate_dim,
            Index output_dim,
            graph::DataType dtype = graph::DataType::FP32);

    //! Constructor: creates GPT-2 MLP where output_dim == input_dim (GPT-2 default)
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension (typically 4 * input_dim)
    //! @param dtype Data type for all tensors
    Gpt2Mlp(graph::NNGraph& graph,
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

    // Submodule accessors (GPT-2 naming)
    module::Linear& c_fc() { return c_fc_; }
    const module::Linear& c_fc() const { return c_fc_; }
    module::Gelu& gelu() { return gelu_; }
    const module::Gelu& gelu() const { return gelu_; }
    module::Linear& c_proj() { return c_proj_; }
    const module::Linear& c_proj() const { return c_proj_; }
};

} // namespace nntile::model::gpt2
