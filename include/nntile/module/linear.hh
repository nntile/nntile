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
class Linear : public Module
{
private:
    // References to parameter tensors (also registered via register_parameter)
    graph::TensorNode* weight_tensor_ = nullptr;
    graph::TensorNode* bias_tensor_ = nullptr;

    // Dimensions and data type
    Index input_dim_;
    Index output_dim_;
    graph::DataType dtype_;

    // Track if forward has been built
    bool forward_built_ = false;

public:
    //! Constructor: creates new weight tensor, no bias
    //! @param graph The logical graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for tensors
    Linear(
        graph::LogicalGraph& graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: creates new weight and optionally bias tensors
    //! @param graph The logical graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param with_bias Whether to create bias tensor
    //! @param dtype Data type for tensors
    Linear(
        graph::LogicalGraph& graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        bool with_bias,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: uses existing weight tensor, no bias
    //! @param graph The logical graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor to use [input_dim, output_dim]
    Linear(
        graph::LogicalGraph& graph,
        const std::string& name,
        graph::TensorNode& weight_tensor
    );

    //! Constructor: uses existing weight and bias tensors
    //! @param graph The logical graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor [input_dim, output_dim]
    //! @param bias_tensor Existing bias tensor [output_dim]
    Linear(
        graph::LogicalGraph& graph,
        const std::string& name,
        graph::TensorNode& weight_tensor,
        graph::TensorNode& bias_tensor
    );

    //! Build forward operation and return output tensor
    //! @param input Input tensor node
    //! @return Reference to the created output tensor
    graph::TensorNode& build_forward(graph::TensorNode& input) override;

    //! Build backward operations using gradient registry
    //!
    //! This method:
    //! 1. Looks up gradient of output tensor from registry
    //! 2. Computes gradient of weight tensor (accumulates if shared)
    //! 3. Computes gradient of bias tensor if present (accumulates if shared)
    //! 4. Computes gradient of input tensor if requires_grad is set
    //! 5. Registers computed gradients in the registry
    //!
    //! @param grad_reg Gradient registry (maps tensors to their gradients)
    //! @throws std::runtime_error if output gradient not found in registry
    //! @throws std::runtime_error if build_forward was not called first
    void build_backward(graph::GradientRegistry& grad_reg) override;

    //! Get string representation with dimensions
    std::string repr() const override;

    // Tensor accessors
    graph::TensorNode* weight_tensor() const { return weight_tensor_; }
    graph::TensorNode* bias_tensor() const { return bias_tensor_; }

    // Check if bias is enabled
    bool has_bias() const { return bias_tensor_ != nullptr; }

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index output_dim() const { return output_dim_; }
    graph::DataType dtype() const { return dtype_; }
};

} // namespace nntile::module