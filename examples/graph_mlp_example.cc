/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/graph_mlp_example.cc
 * Example demonstrating NNTile graph system with MLP.
 *
 * @version 1.1.0
 * */

#include <nntile/context.hh>
#include <nntile/graph.hh>
#include <iostream>
#include <vector>
#include <chrono>

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

    // Define a simple MLP: x -> Linear -> GELU -> Linear -> y
    nntile::graph::LogicalGraph graph("MLP");

    // Input: batch_size x input_dim
    auto& x = graph.tensor(
        nntile::graph::TensorSpec(
            {4, 8}, nntile::graph::DataType::FP32
        ),
        "input"
    );

    // First linear layer: input_dim x hidden_dim
    auto& w1 = graph.tensor(
        nntile::graph::TensorSpec(
            {8, 16}, nntile::graph::DataType::FP32
        ),
        "weight1"
    );

    // Second linear layer: hidden_dim x output_dim
    auto& w2 = graph.tensor(
        nntile::graph::TensorSpec(
            {16, 4}, nntile::graph::DataType::FP32
        ),
        "weight2"
    );

    // Forward pass
    auto& h = nntile::graph::gemm(x, w1, "hidden");
    auto& a = nntile::graph::gelu(h, "activation");
    auto& y = nntile::graph::gemm(a, w2, "output");

    std::cout << "Logical Graph:" << std::endl;
    std::cout << graph.to_string() << std::endl;

    // Compile the graph
    auto compiled = nntile::graph::CompiledGraph::compile(graph);

    // Prepare input data
    std::vector<float> x_data(4 * 8, 1.0f);  // All ones
    std::vector<float> w1_data(8 * 16, 0.1f); // Small weights
    std::vector<float> w2_data(16 * 4, 0.1f); // Small weights

    // Bind data
    compiled.bind_data("input", x_data);
    compiled.bind_data("weight1", w1_data);
    compiled.bind_data("weight2", w2_data);

    // Execute
    auto start = std::chrono::high_resolution_clock::now();
    compiled.execute();
    compiled.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    // Get results
    auto y_data = compiled.get_output<float>("output");

    std::cout << "Execution time: " << duration << " microseconds\n";
    std::cout << "Output shape: [4, 4]\n";
    std::cout << "Sample output values: ";
    for (size_t i = 0; i < std::min(size_t(8), y_data.size()); ++i) {
        std::cout << y_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    return 0;
}
