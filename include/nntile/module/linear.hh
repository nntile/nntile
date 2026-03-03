/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/linear.hh
 * Linear module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Linear module using graph API
//! Adds linear transformation operations to logical graphs
//!
//! Computes: output = input @ weight + bias (bias optional)
//!
//! Supports flexible construction modes:
//! 1. Create new weight/bias tensors (specify dimensions)
//! 2. Use existing weight/bias tensors (for weight/bias sharing)
class Linear : public ModuleBase
{
private:
    // References to parameter tensors (also registered via register_parameter)
    graph::NNGraph::TensorNode* weight_tensor_ = nullptr;
    graph::NNGraph::TensorNode* bias_tensor_ = nullptr;

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

    // Dimensions and data type
    Index input_dim_;
    Index output_dim_;
    graph::DataType dtype_;

public:
    //! Constructor: creates new weight tensor, no bias
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for tensors
    Linear(
        graph::NNGraph& graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: creates new weight and optionally bias tensors
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param with_bias Whether to create bias tensor
    //! @param dtype Data type for tensors
    Linear(
        graph::NNGraph& graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        bool with_bias,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: uses existing weight tensor, no bias
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor to use [input_dim, output_dim]
    Linear(
        graph::NNGraph& graph,
        const std::string& name,
        graph::NNGraph::TensorNode& weight_tensor
    );

    //! Constructor: uses existing weight and bias tensors
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor [input_dim, output_dim]
    //! @param bias_tensor Existing bias tensor [output_dim]
    Linear(
        graph::NNGraph& graph,
        const std::string& name,
        graph::NNGraph::TensorNode& weight_tensor,
        graph::NNGraph::TensorNode& bias_tensor
    );

    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    //! Forward: calls build_forward (user does bookkeeping via autograd ops)
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& input)
    {
        return build_forward(input);
    }

    //! Get string representation with dimensions
    std::string repr() const override;

    // Tensor accessors
    graph::NNGraph::TensorNode* weight_tensor() const { return weight_tensor_; }
    graph::NNGraph::TensorNode* bias_tensor() const { return bias_tensor_; }

    // Check if bias is enabled
    bool has_bias() const { return bias_tensor_ != nullptr; }

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index output_dim() const { return output_dim_; }
    graph::DataType dtype() const { return dtype_; }
};

} // namespace nntile::module
