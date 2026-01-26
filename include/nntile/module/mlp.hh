/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/mlp.hh
 * MLP module implementation using NNTile graph API.
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

namespace nntile::module
{

//! MLP (Multi-Layer Perceptron) module using graph API
//!
//! Architecture: Linear -> GELU -> Linear
//!   - fc1: input_dim -> intermediate_dim
//!   - activation: GELU
//!   - fc2: intermediate_dim -> output_dim
//!
//! This module demonstrates composing multiple submodules.
class Mlp : public Module
{
private:
    //! First linear layer: input -> intermediate
    Linear fc1_;

    //! Activation module: hidden -> activation
    Gelu gelu_;

    //! Second linear layer: activation -> output
    Linear fc2_;

    //! Dimensions
    Index input_dim_;
    Index intermediate_dim_;
    Index output_dim_;
    graph::DataType dtype_;

    //! Intermediate tensors (created during build_forward)
    graph::NNGraph::TensorNode* hidden_tensor_ = nullptr;      // After fc1
    graph::NNGraph::TensorNode* activation_tensor_ = nullptr;  // After GELU

public:
    //! Constructor: creates MLP with specified dimensions
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension (after fc1)
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for all tensors
    Mlp(graph::NNGraph& graph,
        const std::string& name,
        Index input_dim,
        Index intermediate_dim,
        Index output_dim,
        graph::DataType dtype = graph::DataType::FP32);

    //! Constructor: creates MLP where output_dim == input_dim (common in transformers)
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param dtype Data type for all tensors
    Mlp(graph::NNGraph& graph,
        const std::string& name,
        Index input_dim,
        Index intermediate_dim,
        graph::DataType dtype = graph::DataType::FP32);

    //! Build forward operations
    //! @param input Input tensor node
    //! @return Reference to output tensor
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input) override;

    //! Build backward operations using grad fields on NNGraph::TensorNode
    void build_backward() override;

    //! Get string representation with dimensions
    std::string repr() const override;

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index intermediate_dim() const { return intermediate_dim_; }
    Index output_dim() const { return output_dim_; }

    // Submodule accessors
    Linear& fc1() { return fc1_; }
    const Linear& fc1() const { return fc1_; }
    Gelu& gelu() { return gelu_; }
    const Gelu& gelu() const { return gelu_; }
    Linear& fc2() { return fc2_; }
    const Linear& fc2() const { return fc2_; }
};

} // namespace nntile::module
