/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/deep_relu_training.cc
 * Deep ReLU network training example using NNTile graph API with SGD
 * optimizer. Demonstrates forward pass, backward pass (autograd), and
 * optimizer step (under no_grad) in a single compiled graph.
 *
 * Usage:
 *   ./deep_relu_training                       # default settings
 *   ./deep_relu_training <num_iters> <lr>       # custom iterations and LR
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

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;

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

        Index in = input_dim;
        Index out = (depth == 1) ? output_dim : hidden_dim;
        linears_.push_back(std::make_unique<Linear>(
            graph, name + "_linear_0", in, out, dtype));
        register_module("linear_0", linears_.back().get());

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

int main(int argc, char** argv)
{
    const Index input_dim  = 128;
    const Index hidden_dim = 256;
    const Index output_dim = 10;
    const Index depth      = 5;
    const Index batch_size = 32;

    int num_iters = 50;
    float learning_rate = 0.01f;

    if(argc > 1) num_iters = std::atoi(argv[1]);
    if(argc > 2) learning_rate = std::atof(argv[2]);

    std::cout << "=== Deep ReLU Training Example ===\n"
              << "Architecture: " << input_dim << " -> "
              << hidden_dim << " (x" << (depth - 1) << " hidden) -> "
              << output_dim << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Training iterations: " << num_iters << "\n"
              << "Learning rate: " << learning_rate << "\n\n";

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0,
                    "localhost", 5001, 0);

    // ---- Build the computation graph ----
    NNGraph graph("deep_relu_training");

    DeepReLU model(&graph, "net", input_dim, hidden_dim, output_dim,
                   depth, DataType::FP32);

    auto* input = graph.tensor(
        {batch_size, input_dim}, "input", DataType::FP32, false);
    input->mark_input(true);

    auto* target = graph.tensor(
        {batch_size, output_dim}, "target", DataType::FP32, false);
    target->mark_input(true);

    auto* output = model.forward(input);

    // residual = output - target
    auto* residual = add(1.0, output, -1.0, target, "residual");

    // loss = mean((output - target)^2)
    Scalar loss_scale = 1.0 / static_cast<Scalar>(batch_size * output_dim);
    auto* loss = mse_loss(residual, "loss", loss_scale);
    loss->mark_output(true);

    // Set upstream gradient for loss (implicitly 1.0 for scalar loss)
    auto [loss_grad, loss_grad_first] =
        graph.get_or_create_grad(loss, "loss_grad");
    graph::tensor::fill(Scalar(1.0), loss_grad->data());

    // ---- Backward pass ----
    loss->backward(true);

    // ---- Optimizer: SGD under no_grad ----
    // Collect parameter/gradient pairs and create velocity buffers
    struct OptState {
        std::string vel_name;
        Index nelems;
    };
    std::vector<OptState> opt_states;

    auto params = model.named_parameters_recursive();
    for(const auto& [pname, param_tensor] : params)
    {
        auto* param_grad = param_tensor->grad();
        if(param_grad == nullptr)
        {
            continue;
        }

        std::string vel_name = pname + "_velocity";
        auto* velocity = graph.tensor(
            param_tensor->shape(), vel_name,
            param_tensor->dtype(), false);
        velocity->mark_input(true);
        velocity->mark_output(true);

        param_tensor->mark_input(true);
        param_tensor->mark_output(true);

        Index nelems = 1;
        for(auto d : param_tensor->shape()) nelems *= d;

        sgd_step(param_tensor, param_grad, velocity,
                 1, 0.0, learning_rate, 0.0, 0.0, false);

        opt_states.push_back({vel_name, nelems});
    }

    std::cout << "Model structure:\n" << model.to_string() << "\n";

    std::size_t total_params = 0;
    for(const auto& [name, tensor] : params)
    {
        Index n = 1;
        for(auto d : tensor->shape()) n *= d;
        total_params += static_cast<std::size_t>(n);
    }
    std::cout << "Parameters: " << params.size()
              << " tensors, " << total_params << " total values\n\n";

    // ---- Generate random input, target, and initial weights ----
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> input_data(
        static_cast<std::size_t>(batch_size * input_dim));
    for(auto& v : input_data) v = dist(gen);

    // Target: small random values (regression target)
    std::vector<float> target_data(
        static_cast<std::size_t>(batch_size * output_dim));
    std::uniform_real_distribution<float> target_dist(-1.0f, 1.0f);
    for(auto& v : target_data) v = target_dist(gen);

    // Initialize weights with Kaiming uniform
    {
        io::SafeTensorsWriter writer;
        for(const auto& [name, tensor] : params)
        {
            const auto& shape = tensor->shape();
            Index nelems = 1;
            for(auto d : shape) nelems *= d;

            float fan_in = static_cast<float>(shape[0]);
            float limit = std::sqrt(1.0f / fan_in);
            std::uniform_real_distribution<float> wdist(-limit, limit);

            std::vector<float> data(static_cast<std::size_t>(nelems));
            for(auto& v : data) v = wdist(gen);

            std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
            std::memcpy(bytes.data(), data.data(), bytes.size());

            std::vector<std::int64_t> shape64(shape.begin(), shape.end());
            writer.add_tensor(name, tensor->dtype(), shape64, std::move(bytes));
        }
        const std::string weights_path = "/tmp/deep_relu_training_weights.safetensors";
        writer.write(weights_path);
        model.load(weights_path);
    }

    // ---- Compile the graph ----
    TensorGraph::Runtime runtime(graph.tensor_graph());
    runtime.compile();

    // ---- Bind initial data ----
    runtime.bind_data("input", input_data);
    runtime.bind_data("target", target_data);

    // Zero-initialize velocity buffers
    for(const auto& state : opt_states)
    {
        std::vector<float> zeros(static_cast<std::size_t>(state.nelems), 0.0f);
        runtime.bind_data(state.vel_name, zeros);
    }

    // ---- Training loop ----
    std::cout << "Training...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    for(int iter = 0; iter < num_iters; ++iter)
    {
        runtime.execute();
        runtime.wait();

        auto loss_data = runtime.get_output<float>("loss");
        float loss_val = loss_data[0];

        if(iter == 0 || (iter + 1) % 10 == 0 || iter == num_iters - 1)
        {
            std::cout << "  Iter " << (iter + 1) << "/" << num_iters
                      << ": loss = " << loss_val << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_start).count();

    std::cout << "\nTraining completed in " << total_us << " us";
    if(num_iters > 0)
    {
        std::cout << " (" << (total_us / num_iters) << " us/iter)";
    }
    std::cout << "\n";

    std::cout << "\nDone.\n";
    return 0;
}
