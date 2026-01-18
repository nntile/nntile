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

#include <nntile/graph/graph.hh>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    // Initialize StarPU (assuming it's already done in the application)

    // Define a simple MLP: x -> Linear -> GELU -> Linear -> y
    nntile::graph::LogicalGraph graph("MLP");

    // Input: batch_size x input_dim
    auto& x = graph.tensor(nntile::graph::TensorSpec({4, 8}, nntile::graph::DataType::FP32), "input");

    // First linear layer: input_dim x hidden_dim
    auto& w1 = graph.tensor(nntile::graph::TensorSpec({8, 16}, nntile::graph::DataType::FP32), "weight1");

    // Second linear layer: hidden_dim x output_dim
    auto& w2 = graph.tensor(nntile::graph::TensorSpec({16, 4}, nntile::graph::DataType::FP32), "weight2");

    // Forward pass
    auto& h = graph.matmul(x, w1, "hidden");
    auto& a = graph.gelu(h, "activation");
    auto& y = graph.matmul(a, w2, "output");

    // Mark output
    graph.mark_output("output");

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

    // Get results
    auto y_data = compiled.get_output<float>("output");

    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;

    std::cout << "Output shape: [4, 4]" << std::endl;
    std::cout << "Sample output values: ";
    for (size_t i = 0; i < std::min(size_t(8), y_data.size()); ++i) {
        std::cout << y_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    return 0;
}