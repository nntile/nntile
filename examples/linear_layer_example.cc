/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/linear_layer_example.cc
 * Example demonstrating linear module using NNTile graph API.
 *
 * @version 1.1.0
 * */

#include <nntile/context.hh>
#include <nntile/module/linear.hh>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

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

    // Create single logical graph for both forward and backward
    nntile::graph::LogicalGraph graph("Linear_Graph");

    // Create Linear (tied to the graph)
    nntile::module::Linear linear(graph, "linear1", 8, 4, nntile::graph::DataType::FP32);

    // Create input tensor
    auto& input_tensor = graph.tensor(
        nntile::graph::TensorSpec({4, 8}, nntile::graph::DataType::FP32), "external_input");

    // Build forward operation and get output tensor
    auto& output_tensor = linear.build_forward(input_tensor);

    // Create grad tensors in the same graph
    auto& grad_output_tensor = graph.tensor(
        nntile::graph::TensorSpec({4, 4}, nntile::graph::DataType::FP32), "external_grad_output");
    auto& grad_input_tensor = graph.tensor(
        nntile::graph::TensorSpec({4, 8}, nntile::graph::DataType::FP32), "external_grad_input");

    // Build backward operations
    linear.build_backward(grad_output_tensor, grad_input_tensor);

    // Print graph structure for debugging
    std::cout << "Graph structure:" << std::endl;
    std::cout << graph.to_string() << std::endl;

    // Compile the graph
    auto compiled_graph = nntile::graph::CompiledGraph::compile(graph);

    // Generate random input data
    std::vector<float> input_data(4 * 8);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : input_data) {
        val = dist(gen);
    }

    std::cout << "=== Linear Layer Forward/Backward Pass ===" << std::endl;

    // Bind input data to the external input tensor
    compiled_graph.bind_data("external_input", input_data);

    // Initialize weight data
    std::vector<float> weight_data(linear.input_dim() * linear.output_dim());
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::normal_distribution<float> dist2(0.0f, 0.1f);
    for (auto& val : weight_data) {
        val = dist2(gen2);
    }
    compiled_graph.bind_data(linear.weight_name(), weight_data);

    // Initialize gradient data (for backward pass)
    std::vector<float> grad_output_data(4 * 4, 1.0f); // Dummy gradient data
    std::vector<float> grad_input_data(4 * 8, 0.0f);  // Will be filled by backward pass
    compiled_graph.bind_data("external_grad_output", grad_output_data);
    compiled_graph.bind_data("external_grad_input", grad_input_data);

    // Execute the graph (contains both forward and backward operations)
    auto start = std::chrono::high_resolution_clock::now();
    compiled_graph.execute();
    compiled_graph.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Graph execution time: " << duration << " microseconds" << std::endl;
    std::cout << "Input shape: [4, 8]" << std::endl;
    std::cout << "Output shape: [4, 4]" << std::endl;

    // Get output data from the output tensor
    auto output_data = compiled_graph.get_output<float>(linear.output_name());
    std::cout << "Sample output values: ";
    for (size_t i = 0; i < std::min(size_t(8), output_data.size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Get output (this would normally be done after binding input data)
    // For demonstration, we'll just show the graph structure
    std::cout << "\nLinear module created with:" << std::endl;
    std::cout << "- Weight tensor: " << linear.weight_name() << " [8, 4]" << std::endl;
    std::cout << "- Input tensor: external_input [4, 8]" << std::endl;
    std::cout << "- Output tensor: " << linear.output_name() << " [4, 4]" << std::endl;

    std::cout << "\nLinear module successfully created and graphs built!" << std::endl;

    return 0;
}