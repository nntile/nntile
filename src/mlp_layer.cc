/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/mlp_layer.cc
 * MLP layer implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/mlp_layer.hh"

// Include standard headers
#include <algorithm>
#include <random>

// Include NNTile headers
#include <nntile/context.hh>

namespace nntile
{

//! Constructor: only stores dimensions, no tensor construction
MlpLayer::MlpLayer(Index hidden_dim, Index intermediate_dim)
    : hidden_dim_(hidden_dim)
    , intermediate_dim_(intermediate_dim)
    , input_name_("mlp_input")
    , hidden_name_("mlp_hidden")
    , activation_name_("mlp_activation")
    , output_name_("mlp_output")
    , weight1_name_("mlp_weight1")
    , weight2_name_("mlp_weight2")
    , grad_output_name_("mlp_output_grad")
    , grad_activation_name_("mlp_activation_grad")
    , grad_hidden_name_("mlp_hidden_grad")
    , grad_weight1_name_("mlp_weight1_grad")
    , grad_weight2_name_("mlp_weight2_grad")
{
    // Only store dimensions and tensor names, no tensor construction
}

//! Build forward operations in the provided logical graph
void MlpLayer::build_forward_graph(graph::LogicalGraph& graph, const std::vector<Index>& input_shape)
{
    // Extract dimensions: input_shape = [..., input_dim]
    // Leading dimensions are everything except the last dimension
    std::vector<Index> leading_dims(input_shape.begin(), input_shape.end() - 1);
    Index input_dim = input_shape.back();

    // Create parameter tensors (weights) - these are MLP internal
    auto& weight1 = graph.tensor(
        graph::TensorSpec({input_dim, hidden_dim_}, graph::DataType::FP32), weight1_name_);
    auto& weight2 = graph.tensor(
        graph::TensorSpec({hidden_dim_, intermediate_dim_}, graph::DataType::FP32), weight2_name_);

    // Create input tensor (will be bound at runtime)
    auto& input = graph.tensor(
        graph::TensorSpec(input_shape, graph::DataType::FP32), input_name_);

    // Create intermediate activation tensors with same leading dimensions
    std::vector<Index> hidden_shape = leading_dims;
    hidden_shape.push_back(hidden_dim_);

    std::vector<Index> output_shape = leading_dims;
    output_shape.push_back(intermediate_dim_);

    // Linear layer: input[..., input_dim] @ weight1[input_dim, hidden_dim] -> hidden[..., hidden_dim]
    auto& hidden = graph::gemm(input, weight1, hidden_name_, 1.0, false, false, 1, 0);
    auto& activation = graph::gelu(hidden, activation_name_);

    // Linear layer: activation[..., hidden_dim] @ weight2[hidden_dim, intermediate_dim] -> output[..., intermediate_dim]
    auto& output = graph::gemm(activation, weight2, output_name_, 1.0, false, false, 1, 0);
}

//! Build backward operations in the provided logical graph
void MlpLayer::build_backward_graph(graph::LogicalGraph& graph, const std::vector<Index>& input_shape)
{
    // Extract dimensions: input_shape = [..., input_dim]
    std::vector<Index> leading_dims(input_shape.begin(), input_shape.end() - 1);
    Index input_dim = input_shape.back();

    // Create parameter tensors (weights) - these are MLP internal
    auto& weight1 = graph.tensor(
        graph::TensorSpec({input_dim, hidden_dim_}, graph::DataType::FP32), weight1_name_);
    auto& weight2 = graph.tensor(
        graph::TensorSpec({hidden_dim_, intermediate_dim_}, graph::DataType::FP32), weight2_name_);

    // Create forward tensors (for backward dependencies)
    auto& input = graph.tensor(
        graph::TensorSpec(input_shape, graph::DataType::FP32), input_name_);

    std::vector<Index> hidden_shape = leading_dims;
    hidden_shape.push_back(hidden_dim_);
    auto& hidden = graph.tensor(
        graph::TensorSpec(hidden_shape, graph::DataType::FP32), hidden_name_);
    auto& activation = graph.tensor(
        graph::TensorSpec(hidden_shape, graph::DataType::FP32), activation_name_);

    std::vector<Index> output_shape = leading_dims;
    output_shape.push_back(intermediate_dim_);
    auto& output = graph.tensor(
        graph::TensorSpec(output_shape, graph::DataType::FP32), output_name_);

    // Create gradient tensors with same shapes
    auto& grad_output = graph.tensor(
        graph::TensorSpec(output_shape, graph::DataType::FP32), grad_output_name_);
    auto& grad_activation = graph.tensor(
        graph::TensorSpec(hidden_shape, graph::DataType::FP32), grad_activation_name_);
    auto& grad_hidden = graph.tensor(
        graph::TensorSpec(hidden_shape, graph::DataType::FP32), grad_hidden_name_);
    auto& grad_input = graph.tensor(
        graph::TensorSpec(input_shape, graph::DataType::FP32), "mlp_grad_input");

    // Weight gradients
    auto& grad_weight2 = graph.tensor(
        graph::TensorSpec({hidden_dim_, intermediate_dim_}, graph::DataType::FP32), grad_weight2_name_);
    auto& grad_weight1 = graph.tensor(
        graph::TensorSpec({input_dim, hidden_dim_}, graph::DataType::FP32), grad_weight1_name_);

    // Backward pass operations (reverse order)
    // For gradients with respect to weights, we need to sum over all leading dimensions
    // dL/dW2 = sum over leading dims of activation^T @ grad_output
    graph::gemm(activation, grad_output, grad_weight2, 1.0, 0.0, true, false, 1, 0);

    // dL/dA = grad_output @ W2^T (broadcasted over leading dimensions)
    graph::gemm(grad_output, weight2, grad_activation, 1.0, 0.0, false, true, 1, 0);

    // dL/dH = dL/dA * gelu_backward(H)
    graph::gelu_backward(hidden, grad_activation, grad_hidden);

    // dL/dW1 = sum over leading dims of input^T @ grad_hidden
    graph::gemm(input, grad_hidden, grad_weight1, 1.0, 0.0, true, false, 1, 0);

    // dL/dX = grad_hidden @ W1^T (broadcasted over leading dimensions)
    graph::gemm(grad_hidden, weight1, grad_input, 1.0, 0.0, false, true, 1, 0);
}

//! Forward pass: bind data and execute on compiled forward graph
void MlpLayer::forward(graph::CompiledGraph& compiled_forward_graph, const std::vector<float>& input_data, const std::vector<Index>& input_shape)
{
    // Extract input dimension from shape
    Index input_dim = input_shape.back();

    // Bind input data
    compiled_forward_graph.bind_data(input_name_, input_data);

    // Initialize weights if not already done (first forward pass)
    static bool weights_initialized = false;
    if (!weights_initialized) {
        auto weight1_data = init_weights(input_dim * hidden_dim_);
        auto weight2_data = init_weights(hidden_dim_ * intermediate_dim_);
        compiled_forward_graph.bind_data(weight1_name_, weight1_data);
        compiled_forward_graph.bind_data(weight2_name_, weight2_data);
        weights_initialized = true;
    }
}

//! Backward pass: transfer forward results to backward graph and execute
void MlpLayer::backward(graph::CompiledGraph& compiled_forward_graph,
                        graph::CompiledGraph& compiled_backward_graph,
                        const std::vector<float>& output_grad,
                        const std::vector<Index>& input_shape)
{
    // Get forward pass results
    auto input_data = compiled_forward_graph.get_output<float>(input_name_);
    auto hidden_data = compiled_forward_graph.get_output<float>(hidden_name_);
    auto activation_data = compiled_forward_graph.get_output<float>(activation_name_);
    auto output_data = compiled_forward_graph.get_output<float>(output_name_);
    auto weight1_data = compiled_forward_graph.get_output<float>(weight1_name_);
    auto weight2_data = compiled_forward_graph.get_output<float>(weight2_name_);

    // Bind forward results to backward graph
    compiled_backward_graph.bind_data(input_name_, input_data);
    compiled_backward_graph.bind_data(hidden_name_, hidden_data);
    compiled_backward_graph.bind_data(activation_name_, activation_data);
    compiled_backward_graph.bind_data(output_name_, output_data);
    compiled_backward_graph.bind_data(weight1_name_, weight1_data);
    compiled_backward_graph.bind_data(weight2_name_, weight2_data);

    // Bind output gradient
    compiled_backward_graph.bind_data(grad_output_name_, output_grad);

    // Extract dimensions from input_shape
    std::vector<Index> leading_dims(input_shape.begin(), input_shape.end() - 1);
    Index input_dim = input_shape.back();

    // Size calculations: product of all dimensions
    auto compute_size = [](const std::vector<Index>& shape) -> size_t {
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        return size;
    };

    std::vector<Index> hidden_shape = leading_dims;
    hidden_shape.push_back(hidden_dim_);

    std::vector<Index> output_shape = leading_dims;
    output_shape.push_back(intermediate_dim_);

    std::vector<float> zero_grad_hidden(compute_size(hidden_shape), 0.0f);
    std::vector<float> zero_grad_activation(compute_size(hidden_shape), 0.0f);
    std::vector<float> zero_grad_input(compute_size(input_shape), 0.0f);
    std::vector<float> zero_grad_weight1(input_dim * hidden_dim_, 0.0f);
    std::vector<float> zero_grad_weight2(hidden_dim_ * intermediate_dim_, 0.0f);

    compiled_backward_graph.bind_data(grad_hidden_name_, zero_grad_hidden);
    compiled_backward_graph.bind_data(grad_activation_name_, zero_grad_activation);
    compiled_backward_graph.bind_data("mlp_grad_input", zero_grad_input);
    compiled_backward_graph.bind_data(grad_weight1_name_, zero_grad_weight1);
    compiled_backward_graph.bind_data(grad_weight2_name_, zero_grad_weight2);
}

// Accessors for parameter gradients (from backward graph)
std::vector<float> MlpLayer::get_weight1_grad(graph::CompiledGraph& compiled_backward_graph) const
{
    return compiled_backward_graph.get_output<float>(grad_weight1_name_);
}

std::vector<float> MlpLayer::get_weight2_grad(graph::CompiledGraph& compiled_backward_graph) const
{
    return compiled_backward_graph.get_output<float>(grad_weight2_name_);
}

// Accessors for parameters (from forward graph)
std::vector<float> MlpLayer::get_weight1(graph::CompiledGraph& compiled_forward_graph) const
{
    return compiled_forward_graph.get_output<float>(weight1_name_);
}

std::vector<float> MlpLayer::get_weight2(graph::CompiledGraph& compiled_forward_graph) const
{
    return compiled_forward_graph.get_output<float>(weight2_name_);
}

// Setters for parameters (to forward graph)
void MlpLayer::set_weight1(graph::CompiledGraph& compiled_forward_graph, const std::vector<float>& data)
{
    compiled_forward_graph.bind_data(weight1_name_, data);
}

void MlpLayer::set_weight2(graph::CompiledGraph& compiled_forward_graph, const std::vector<float>& data)
{
    compiled_forward_graph.bind_data(weight2_name_, data);
}

// Get output data (from forward graph)
std::vector<float> MlpLayer::get_output(graph::CompiledGraph& compiled_forward_graph) const
{
    return compiled_forward_graph.get_output<float>(output_name_);
}

// Get input gradient (from backward graph)
std::vector<float> MlpLayer::get_input_grad(graph::CompiledGraph& compiled_backward_graph) const
{
    return compiled_backward_graph.get_output<float>("mlp_grad_input");
}

// Helper to initialize weights with small random values
std::vector<float> MlpLayer::init_weights(size_t size) const
{
    std::vector<float> weights(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (size_t i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }

    return weights;
}

} // namespace nntile