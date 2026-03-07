/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/module/mlp.hh
 * MLP module implementation using NNTile graph API.
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
#include <nntile/graph/nn.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::module
{

//! MLP (Multi-Layer Perceptron) module using graph API
//!
//! Architecture: Linear -> Activation -> Linear
//!   - fc1: input_dim -> intermediate_dim
//!   - activation: gelu (default), gelutanh, relu, or silu
//!   - fc2: intermediate_dim -> output_dim
//!
//! This module demonstrates composing multiple submodules.
class Mlp : public Module
{
private:
    //! First linear layer: input -> intermediate
    Linear fc1_;

    //! Activation module: hidden -> activation
    Activation activation_;

    //! Second linear layer: activation -> output
    Linear fc2_;

    //! Dimensions
    Index input_dim_;
    Index intermediate_dim_;
    Index output_dim_;
    DataType dtype_;

    //! Intermediate tensors (created during forward)
    NNGraph::TensorNode* hidden_tensor_ = nullptr;      // After fc1
    NNGraph::TensorNode* activation_tensor_ = nullptr;  // After activation

    NNGraph::TensorNode* input_tensor_ = nullptr;
    NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor: creates MLP with specified dimensions
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input feature dimension
    //! @param intermediate_dim Hidden layer dimension (after fc1)
    //! @param output_dim Output feature dimension
    //! @param activation Activation function (default: gelu)
    //! @param dtype Data type for all tensors
    Mlp(NNGraph* graph,
        const std::string& name,
        Index input_dim,
        Index intermediate_dim,
        Index output_dim,
        ActivationType activation = ActivationType::GELU,
        DataType dtype = DataType::FP32);

    //! Constructor: creates MLP where output_dim == input_dim (common in transformers)
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param input_dim Input/output feature dimension
    //! @param intermediate_dim Hidden layer dimension
    //! @param activation Activation function (default: gelu)
    //! @param dtype Data type for all tensors
    Mlp(NNGraph* graph,
        const std::string& name,
        Index input_dim,
        Index intermediate_dim,
        ActivationType activation = ActivationType::GELU,
        DataType dtype = DataType::FP32);

#ifdef NNTILE_HAVE_TORCH
    //! Constructor: creates MLP from PyTorch Linear layers (fc1, fc2) with
    //! automatic weight/bias binding for easy data transfer.
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name
    //! @param fc1_layer PyTorch first linear layer (input -> intermediate)
    //! @param fc2_layer PyTorch second linear layer (intermediate -> output)
    //! @param activation Activation function (must match PyTorch MLP)
    //! @param dtype Data type for all tensors
    Mlp(NNGraph* graph,
        const std::string& name,
        const torch::nn::Linear& fc1_layer,
        const torch::nn::Linear& fc2_layer,
        ActivationType activation = ActivationType::GELU,
        DataType dtype = DataType::FP32);
#endif

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input);

    //! Forward: calls forward
    NNGraph::TensorNode* operator()(NNGraph::TensorNode* input)
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
    Linear& fc1() { return fc1_; }
    const Linear& fc1() const { return fc1_; }
    Activation& activation() { return activation_; }
    const Activation& activation() const { return activation_; }
    Linear& fc2() { return fc2_; }
    const Linear& fc2() const { return fc2_; }
};

#ifdef NNTILE_HAVE_TORCH

inline Mlp::Mlp(NNGraph* graph,
                const std::string& name,
                const torch::nn::Linear& fc1_layer,
                const torch::nn::Linear& fc2_layer,
                ActivationType activation,
                DataType dtype)
    : Module(graph, name)
    , fc1_(graph, name + "_fc1", fc1_layer, dtype)
    , activation_(graph, name + "_activation", activation)
    , fc2_(graph, name + "_fc2", fc2_layer, dtype)
    , input_dim_(static_cast<Index>(fc1_layer->weight.size(1)))
    , intermediate_dim_(static_cast<Index>(fc1_layer->weight.size(0)))
    , output_dim_(static_cast<Index>(fc2_layer->weight.size(0)))
    , dtype_(dtype)
{
    // Validate fc2 input dim matches fc1 output dim
    if(static_cast<Index>(fc2_layer->weight.size(1)) != intermediate_dim_)
    {
        throw std::invalid_argument(
            "Mlp::Mlp: fc2 input dimension must match fc1 output dimension. "
            "fc1 out=" + std::to_string(intermediate_dim_) +
            ", fc2 in=" + std::to_string(fc2_layer->weight.size(1)));
    }
    register_module("fc1", &fc1_);
    register_module("activation", &activation_);
    register_module("fc2", &fc2_);
}

#endif // NNTILE_HAVE_TORCH

} // namespace nntile::graph::module
