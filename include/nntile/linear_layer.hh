/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/linear_layer.hh
 * Linear layer implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include NNTile headers
#include <nntile/graph.hh>

namespace nntile
{

//! Linear layer using graph API - adds linear transformation operations to logical graphs
class LinearLayer
{
private:
    // Reference to the graph this layer belongs to
    graph::LogicalGraph& graph_;

    // Layer name for generating unique tensor names
    std::string name_;

    // References to tensors created during build_forward
    graph::TensorNode* weight_tensor_;
    graph::TensorNode* input_tensor_;
    graph::TensorNode* output_tensor_;

    // Dimensions and data type
    Index input_dim_;
    Index output_dim_;
    graph::DataType dtype_;

public:
    //! Constructor: stores graph reference, name and dimensions
    //! @param graph The logical graph this layer belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for the weight tensor
    LinearLayer(graph::LogicalGraph& graph, const std::string& name, Index input_dim, Index output_dim,
                graph::DataType dtype = graph::DataType::FP32);

    //! Build forward operation and return output tensor
    //! @param input_tensor Input tensor node
    //! @return Reference to the created output tensor
    graph::TensorNode& build_forward(graph::TensorNode& input_tensor);

    //! Build backward operation
    //! @param grad_output_tensor Gradient w.r.t. output
    //! @param grad_input_tensor Gradient w.r.t. input (modified in place)
    void build_backward(graph::TensorNode& grad_output_tensor, graph::TensorNode& grad_input_tensor);

    // Tensor name accessors (generated from layer name)
    std::string input_name() const { return name_ + "_input"; }
    std::string weight_name() const { return name_ + "_weight"; }
    std::string output_name() const { return name_ + "_output"; }
    std::string grad_output_name() const { return name_ + "_output_grad"; }
    std::string grad_weight_name() const { return name_ + "_weight_grad"; }
    std::string grad_input_name() const { return name_ + "_input_grad"; }

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index output_dim() const { return output_dim_; }
};

} // namespace nntile