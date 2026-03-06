/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/deep_relu_forward.cc
 * Deep ReLU network that loads weights from a SafeTensors file and runs
 * a forward pass. Demonstrates NNTile-native serialization.
 *
 * Usage:
 *   ./deep_relu_forward                     # generate weights, save, load, run
 *   ./deep_relu_forward model.safetensors   # load existing weights, run
 *
 * @version 1.1.0
 * */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nntile.hh>
#include <nntile/io/safetensors.hh>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

// A deep ReLU network: a chain of (Linear -> ReLU) blocks, built by
// composing Module primitives. The last layer has no activation.
//
//   input -> [Linear -> ReLU] x (depth-1) -> Linear -> output
//
class DeepReLU : public Module
{
    std::vector<std::unique_ptr<Linear>> linears_;
    std::vector<std::unique_ptr<Activation>> activations_;
    Index depth_;

public:
    DeepReLU(NNGraph* graph,
             const std::string& name,
             Index input_dim,
             Index hidden_dim,
             Index output_dim,
             Index depth,
             DataType dtype = DataType::FP32)
        : Module(graph, name)
        , depth_(depth)
    {
        if(depth < 1)
        {
            throw std::invalid_argument("DeepReLU: depth must be >= 1");
        }

        // First layer
        Index in = input_dim;
        Index out = (depth == 1) ? output_dim : hidden_dim;
        linears_.push_back(std::make_unique<Linear>(
            graph, name + "_linear_0", in, out, dtype));
        register_module("linear_0", linears_.back().get());

        // Hidden layers with ReLU
        for(Index i = 1; i < depth; ++i)
        {
            activations_.push_back(std::make_unique<Activation>(
                graph, name + "_relu_" + std::to_string(i - 1),
                ActivationType::RELU));
            register_module(
                "relu_" + std::to_string(i - 1), activations_.back().get());

            in = hidden_dim;
            out = (i == depth - 1) ? output_dim : hidden_dim;
            linears_.push_back(std::make_unique<Linear>(
                graph, name + "_linear_" + std::to_string(i),
                in, out, dtype));
            register_module(
                "linear_" + std::to_string(i), linears_.back().get());
        }
    }

    NNGraph::TensorNode* forward(NNGraph::TensorNode* x)
    {
        x = linears_[0]->forward(x);
        for(Index i = 1; i < depth_; ++i)
        {
            x = activations_[static_cast<std::size_t>(i - 1)]->forward(x);
            x = linears_[static_cast<std::size_t>(i)]->forward(x);
        }
        return x;
    }

    std::string repr() const override
    {
        return "DeepReLU(depth=" + std::to_string(depth_) + ")";
    }

    Index depth() const { return depth_; }
    Linear& linear(Index i) { return *linears_.at(static_cast<std::size_t>(i)); }
};

// Generate Kaiming-uniform-style random weights and save to SafeTensors.
static void generate_and_save_weights(
    const std::string& path,
    DeepReLU& model,
    unsigned seed = 42)
{
    std::mt19937 gen(seed);

    io::SafeTensorsWriter writer;
    auto params = model.named_parameters_recursive();

    for(const auto& [name, tensor] : params)
    {
        const auto& shape = tensor->shape();
        Index nelems = 1;
        for(auto d : shape) nelems *= d;

        // Kaiming uniform: fan_in = shape[0] for weight
        float fan_in = static_cast<float>(shape[0]);
        float limit = std::sqrt(1.0f / fan_in);
        std::uniform_real_distribution<float> dist(-limit, limit);

        std::vector<float> data(static_cast<std::size_t>(nelems));
        for(auto& v : data) v = dist(gen);

        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());

        std::vector<std::int64_t> shape64(shape.begin(), shape.end());
        writer.add_tensor(name, tensor->dtype(), shape64, std::move(bytes));
    }

    writer.write(path);
}

int main(int argc, char** argv)
{
    // ---- Configuration ----
    const Index input_dim  = 128;
    const Index hidden_dim = 256;
    const Index output_dim = 10;
    const Index depth      = 5;    // 5 linear layers, 4 ReLUs
    const Index batch_size = 32;
    const std::string default_path = "/tmp/deep_relu_weights.safetensors";

    const std::string weights_path =
        (argc > 1) ? argv[1] : default_path;

    // ---- StarPU context ----
    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0,
                    "localhost", 5001, 0);

    // ---- Build the graph ----
    NNGraph graph("deep_relu");

    DeepReLU model(&graph, "net", input_dim, hidden_dim, output_dim,
                   depth, DataType::FP32);

    auto* input = graph.tensor(
        {batch_size, input_dim}, "input", DataType::FP32, false);
    input->mark_input(true);

    auto* output = model.forward(input);
    output->mark_output(true);

    std::cout << "Model structure:\n" << model.to_string() << "\n";

    // Count parameters
    auto params = model.named_parameters_recursive();
    std::size_t total_params = 0;
    for(const auto& [name, tensor] : params)
    {
        Index n = 1;
        for(auto d : tensor->shape()) n *= d;
        total_params += static_cast<std::size_t>(n);
    }
    std::cout << "Parameters: " << params.size()
              << " tensors, " << total_params << " total values\n";

    // ---- Generate weights if no file provided ----
    if(argc <= 1)
    {
        std::cout << "\nGenerating random weights -> " << weights_path << "\n";
        generate_and_save_weights(weights_path, model);
    }

    // ---- Load weights from SafeTensors ----
    std::cout << "Loading weights from " << weights_path << "\n";
    model.load(weights_path);

    // ---- Compile ----
    TensorGraph::Runtime runtime(graph.tensor_graph());
    runtime.compile();

    // ---- Prepare input ----
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input_data(
        static_cast<std::size_t>(batch_size * input_dim));
    for(auto& v : input_data) v = dist(gen);

    runtime.bind_data("input", input_data);

    // ---- Execute forward pass ----
    auto t0 = std::chrono::high_resolution_clock::now();
    runtime.execute();
    runtime.wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();
    std::cout << "\nForward pass: " << us << " us\n";

    // ---- Print output ----
    auto out_data = runtime.get_output<float>(output->name());
    std::cout << "Output shape: [" << batch_size << ", " << output_dim << "]\n";

    // Show first few output values (first sample, all outputs)
    std::cout << "Output[0, :] = [";
    for(Index j = 0; j < output_dim; ++j)
    {
        if(j > 0) std::cout << ", ";
        // Column-major: element [b, j] at offset b + j * batch_size
        std::cout << out_data[static_cast<std::size_t>(0 + j * batch_size)];
    }
    std::cout << "]\n";

    // ---- Verify the file is readable ----
    io::SafeTensorsReader reader(weights_path);
    std::cout << "\nSafeTensors file contains " << reader.size()
              << " tensors:\n";
    for(const auto& tname : reader.tensor_names())
    {
        const auto& info = reader.tensor_info(tname);
        std::cout << "  " << tname << "  dtype="
                  << io::dtype_to_safetensors(info.dtype) << "  shape=[";
        for(std::size_t i = 0; i < info.shape.size(); ++i)
        {
            if(i > 0) std::cout << ", ";
            std::cout << info.shape[i];
        }
        std::cout << "]  " << info.data_size << " bytes\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
