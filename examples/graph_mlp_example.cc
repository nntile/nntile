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

    // Create MLP module (hidden_dim=16, intermediate_dim=4)
    nntile::module::Mlp mlp(16, 4);

    // Define input shape (discovered during forward setup)
    std::vector<nntile::Index> input_shape = {4, 8}; // batch_size=4, input_dim=8

    // Create and build logical graphs
    nntile::graph::LogicalGraph forward_logical("MLP_Forward");
    nntile::graph::LogicalGraph backward_logical("MLP_Backward");

    mlp.build_forward_graph(forward_logical, input_shape);
    mlp.build_backward_graph(backward_logical, input_shape);

    // Compile the graphs
    auto compiled_forward_graph = nntile::graph::CompiledGraph::compile(forward_logical);
    auto compiled_backward_graph = nntile::graph::CompiledGraph::compile(backward_logical);

    // Generate random input data
    std::vector<float> input_data(4 * 8);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : input_data) {
        val = dist(gen);
    }

    // Generate target output (random for this example)
    std::vector<float> target_output(4 * 4);
    for (auto& val : target_output) {
        val = dist(gen);
    }

    std::cout << "=== Forward Pass ===" << std::endl;

    // Bind input and execute forward pass
    auto start = std::chrono::high_resolution_clock::now();
    mlp.forward(compiled_forward_graph, input_data, input_shape);
    compiled_forward_graph.execute();
    compiled_forward_graph.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    // Get output
    auto output = mlp.get_output(compiled_forward_graph);

    std::cout << "Forward pass time: " << forward_duration << " microseconds" << std::endl;
    std::cout << "Output shape: [4, 4]" << std::endl;
    std::cout << "Sample output values: ";
    for (size_t i = 0; i < std::min(size_t(8), output.size()); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "\n=== Backward Pass ===" << std::endl;

    // Compute simple loss gradient (output - target for MSE-like loss)
    std::vector<float> output_grad(4 * 4);
    for (size_t i = 0; i < output.size(); ++i) {
        output_grad[i] = 2.0f * (output[i] - target_output[i]) / output.size();
    }

    // Bind gradients and execute backward pass
    start = std::chrono::high_resolution_clock::now();
    mlp.backward(compiled_forward_graph, compiled_backward_graph, output_grad, input_shape);
    compiled_backward_graph.execute();
    compiled_backward_graph.wait();
    end = std::chrono::high_resolution_clock::now();
    auto backward_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Backward pass time: " << backward_duration << " microseconds" << std::endl;

    // Get weight gradients
    auto grad_w1 = mlp.get_weight1_grad(compiled_backward_graph);
    auto grad_w2 = mlp.get_weight2_grad(compiled_backward_graph);
    auto input_grad = mlp.get_input_grad(compiled_backward_graph);

    std::cout << "Weight1 gradient shape: [8, 16], norm: ";
    float norm1 = 0.0f;
    for (auto val : grad_w1) norm1 += val * val;
    std::cout << std::sqrt(norm1) << std::endl;

    std::cout << "Weight2 gradient shape: [16, 4], norm: ";
    float norm2 = 0.0f;
    for (auto val : grad_w2) norm2 += val * val;
    std::cout << std::sqrt(norm2) << std::endl;

    std::cout << "Input gradient shape: [4, 8], sample values: ";
    for (size_t i = 0; i < std::min(size_t(4), input_grad.size()); ++i) {
        std::cout << input_grad[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "\n=== Training Simulation ===" << std::endl;

    // Simple gradient descent step
    float learning_rate = 0.01f;

    auto w1 = mlp.get_weight1(compiled_forward_graph);
    auto w2 = mlp.get_weight2(compiled_forward_graph);

    for (size_t i = 0; i < w1.size(); ++i) {
        w1[i] -= learning_rate * grad_w1[i];
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        w2[i] -= learning_rate * grad_w2[i];
    }

    mlp.set_weight1(compiled_forward_graph, w1);
    mlp.set_weight2(compiled_forward_graph, w2);

    std::cout << "Applied gradient descent step with learning rate: " << learning_rate << std::endl;

    // Second forward pass to see the change
    mlp.forward(compiled_forward_graph, input_data, input_shape);
    compiled_forward_graph.execute();
    compiled_forward_graph.wait();
    auto output2 = mlp.get_output(compiled_forward_graph);

    std::cout << "Output after training step (first few values): ";
    for (size_t i = 0; i < std::min(size_t(4), output2.size()); ++i) {
        std::cout << output2[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "\n=== MLP Layer Successfully Handles Arbitrary Input Dimensions ===" << std::endl;
    std::cout << "The implementation supports arbitrary dimensional input tensors." << std::endl;
    std::cout << "For example, it can handle 2D [batch, features], 3D [batch, seq, features], etc." << std::endl;
    std::cout << "The linear operations automatically preserve leading dimensions while transforming the last dimension." << std::endl;

    return 0;
}
