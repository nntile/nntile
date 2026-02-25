/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/graph_mlp_example.cc
 * Example demonstrating trainable MLP module using NNTile graph API.
 *
 * @version 1.1.0
 * */

#include <nntile/context.hh>
#include <nntile/module/mlp.hh>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    // Initialize NNTile context (this initializes StarPU)
    nntile::Context context(
        1, // ncpu: number of CPU workers
        0, // ncuda: number of CUDA workers
        0, // ooc: enable Out-of-Core (0=disabled)
        "/tmp/nntile_ooc", // ooc_path: path for OOC disk
        16777216, // ooc_size: OOC disk size in bytes
        0, // logger: enable logger (0=disabled)
        "localhost", // logger_addr: logger server address
        5001, // logger_port: logger server port
        0 // verbose: verbosity level (0=quiet)
    );

    // Create a neural network graph for forward and backward
    nntile::graph::NNGraph graph("MLP_Graph");

    // Create MLP module (input_dim=8, intermediate_dim=16, output_dim=4)
    nntile::module::Mlp mlp(
        graph, "mlp", 8, 16, 4, nntile::graph::DataType::FP32);

    // Create input tensor (requires_grad to compute input gradients)
    // Shape is [batch, features] (last dim = features): 4 batches, 8 features
    auto& input_tensor = graph.tensor(
        {4, 8},
        "external_input",
        nntile::graph::DataType::FP32,
        true);
    input_tensor.mark_input(true);  // bind_data() requires input marking

    // Build forward operation and get output tensor
    auto& output_tensor = mlp.build_forward(input_tensor);
    output_tensor.mark_output(true);  // get_output() requires output marking

    // Attach an external gradient to the output (e.g., loss gradient)
    auto& grad_output_tensor = graph.get_or_create_grad(
        output_tensor,
        "external_grad_output");
    grad_output_tensor.mark_input(true);  // bind_data() requires input marking

    // Mark parameter tensors for bind_data (weights)
    mlp.fc1().weight_tensor()->mark_input(true);
    mlp.fc2().weight_tensor()->mark_input(true);

    // Build backward operations
    mlp.build_backward();

    // Mark gradient tensors for get_output (created during build_backward)
    mlp.fc1().weight_tensor()->grad()->mark_output(true);
    mlp.fc2().weight_tensor()->grad()->mark_output(true);
    if (input_tensor.has_grad()) {
        input_tensor.grad()->mark_output(true);
    }

    // Print graph structure for debugging
    std::cout << "Graph structure:" << std::endl;
    std::cout << graph.to_string() << std::endl;

    // Compile the graph
    auto compiled_graph = nntile::graph::CompiledGraph::compile(
        graph.logical_graph());

    // Generate random input data (4 batches x 8 features)
    std::vector<float> input_data(4 * 8);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : input_data) {
        val = dist(gen);
    }

    // Bind input data to the external input tensor
    compiled_graph.bind_data("external_input", input_data);

    // Initialize weights (reuse gen from input data generation)
    std::vector<float> w1_data(8 * 16);
    std::vector<float> w2_data(16 * 4);
    std::normal_distribution<float> dist2(0.0f, 0.1f);
    for (auto& val : w1_data) {
        val = dist2(gen);
    }
    for (auto& val : w2_data) {
        val = dist2(gen);
    }

    compiled_graph.bind_data(mlp.fc1().weight_tensor()->name(), w1_data);
    compiled_graph.bind_data(mlp.fc2().weight_tensor()->name(), w2_data);

    // Initialize gradient data (for backward pass): 4 batches x 4 output features
    std::vector<float> grad_output_data(4 * 4, 1.0f);
    compiled_graph.bind_data(grad_output_tensor.name(), grad_output_data);

    std::cout << "=== MLP Forward/Backward Pass ===" << std::endl;

    // Execute the graph (contains both forward and backward operations)
    auto start = std::chrono::high_resolution_clock::now();
    compiled_graph.execute();
    compiled_graph.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Graph execution time: " << duration << " microseconds" << std::endl;

    // Get output data
    auto output_data = compiled_graph.get_output<float>(output_tensor.name());
    std::cout << "Sample output values: ";
    for (size_t i = 0; i < std::min(size_t(8), output_data.size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Get gradients
    auto grad_w1 = compiled_graph.get_output<float>(
        mlp.fc1().grad_name("weight"));
    auto grad_w2 = compiled_graph.get_output<float>(
        mlp.fc2().grad_name("weight"));
    std::cout << "Weight1 grad size: " << grad_w1.size() << std::endl;
    std::cout << "Weight2 grad size: " << grad_w2.size() << std::endl;
    if (input_tensor.has_grad()) {
        auto grad_input = compiled_graph.get_output<float>(
            input_tensor.grad()->name());
        std::cout << "Input grad size: " << grad_input.size() << std::endl;
    } else {
        std::cout << "Input grad not available." << std::endl;
    }

    std::cout << "\nMLP module successfully created and graphs built!" << std::endl;

    return 0;
}
