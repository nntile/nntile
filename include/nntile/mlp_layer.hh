/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/mlp_layer.hh
 * MLP layer implementation using NNTile graph API.
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

namespace nntile
{

//! MLP layer using graph API - builds operations into provided logical graphs
class MlpLayer
{
private:
    // Tensor names for easy access
    std::string input_name_;
    std::string hidden_name_;
    std::string activation_name_;
    std::string output_name_;
    std::string weight1_name_;
    std::string weight2_name_;
    std::string grad_output_name_;
    std::string grad_activation_name_;
    std::string grad_hidden_name_;
    std::string grad_weight1_name_;
    std::string grad_weight2_name_;

    // Tensor shapes (input shape discovered during build)
    Index batch_size_;
    Index hidden_dim_;
    Index intermediate_dim_;

public:
    //! Constructor: only stores dimensions, no tensor construction
    //! @param hidden_dim Hidden/intermediate dimension
    //! @param intermediate_dim Output dimension (intermediate for next layer)
    MlpLayer(Index hidden_dim, Index intermediate_dim);

    //! Build forward operations in the provided logical graph
    //! @param graph The logical graph to add operations to
    //! @param input_shape Shape of input tensor [batch_size, input_dim]
    void build_forward_graph(graph::LogicalGraph& graph, const std::vector<Index>& input_shape);

    //! Build backward operations in the provided logical graph
    //! @param graph The logical graph to add operations to
    //! @param input_shape Shape of input tensor [batch_size, input_dim]
    void build_backward_graph(graph::LogicalGraph& graph, const std::vector<Index>& input_shape);

    //! Forward pass: bind data and execute on compiled graph
    //! @param compiled_graph Compiled graph containing this layer's operations
    //! @param input_data Input data vector
    void forward(graph::CompiledGraph& compiled_graph, const std::vector<float>& input_data, const std::vector<Index>& input_shape);

    //! Backward pass: transfer forward results and execute backward
    //! @param compiled_forward_graph Compiled forward graph with results
    //! @param compiled_backward_graph Compiled backward graph to execute
    //! @param output_grad Gradient w.r.t. output
    //! @param input_shape Shape of input tensor [..., input_dim]
    void backward(graph::CompiledGraph& compiled_forward_graph,
                  graph::CompiledGraph& compiled_backward_graph,
                  const std::vector<float>& output_grad,
                  const std::vector<Index>& input_shape);

    // Accessors for parameter gradients (for optimizer)
    std::vector<float> get_weight1_grad(graph::CompiledGraph& compiled_graph) const;
    std::vector<float> get_weight2_grad(graph::CompiledGraph& compiled_graph) const;

    // Accessors for parameters (for optimizer)
    std::vector<float> get_weight1(graph::CompiledGraph& compiled_graph) const;
    std::vector<float> get_weight2(graph::CompiledGraph& compiled_graph) const;

    // Setters for parameters (for optimizer)
    void set_weight1(graph::CompiledGraph& compiled_graph, const std::vector<float>& data);
    void set_weight2(graph::CompiledGraph& compiled_graph, const std::vector<float>& data);

    // Get output data
    std::vector<float> get_output(graph::CompiledGraph& compiled_graph) const;

    // Get input gradient
    std::vector<float> get_input_grad(graph::CompiledGraph& compiled_graph) const;

    // Tensor name accessors
    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }
    const std::string& weight1_name() const { return weight1_name_; }
    const std::string& weight2_name() const { return weight2_name_; }

private:
    // Helper to initialize weights with small random values
    std::vector<float> init_weights(size_t size) const;
};

} // namespace nntile