/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/deep_relu_training.cc
 * Deep ReLU network training example using NNTile graph API with optimizer
 * classes (SGD, Adam, AdamW). Demonstrates forward pass, backward pass
 * (autograd), and optimizer step in a single compiled graph.
 *
 * Supports saving/loading model weights and optimizer state to/from
 * SafeTensors files, plus optimizer config in JSON.
 *
 * Usage:
 *   ./deep_relu_training [options]
 *
 * Options:
 *   --iters N            Number of training iterations (default: 50)
 *   --lr RATE            Learning rate (default: 0.01)
 *   --optimizer TYPE     Optimizer: sgd, adam, adamw (default: sgd)
 *   --momentum VAL       SGD momentum (default: 0.0)
 *   --weight-decay VAL   Weight decay (default: 0.0)
 *   --beta1 VAL          Adam/AdamW beta_1 (default: 0.9)
 *   --beta2 VAL          Adam/AdamW beta_2 (default: 0.999)
 *   --save-model PATH    Save model weights after training
 *   --load-model PATH    Load model weights before training
 *   --save-optim PATH    Save optimizer state after training
 *   --load-optim PATH    Load optimizer state before training
 *   --save-config PATH   Save optimizer config (JSON) after training
 *   --load-config PATH   Load optimizer config (JSON) before training
 *   --export-hf PATH     Export model weights in HF SafeTensors format
 *   --import-hf PATH     Import model weights from HF SafeTensors format
 *   --export-hf-optim PATH  Export optimizer state in HF SafeTensors format
 *   --import-hf-optim PATH  Import optimizer state from HF SafeTensors format
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
using namespace nntile::graph::optim;

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

struct Args
{
    int num_iters = 50;
    float learning_rate = 0.01f;
    std::string optimizer_type = "sgd";
    float momentum = 0.0f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    std::string save_model;
    std::string load_model;
    std::string save_optim;
    std::string load_optim;
    std::string save_config;
    std::string load_config;
    std::string export_hf;
    std::string import_hf;
    std::string export_hf_optim;
    std::string import_hf_optim;
};

Args parse_args(int argc, char** argv)
{
    Args args;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--iters" && i + 1 < argc)
            args.num_iters = std::atoi(argv[++i]);
        else if(arg == "--lr" && i + 1 < argc)
            args.learning_rate = static_cast<float>(std::atof(argv[++i]));
        else if(arg == "--optimizer" && i + 1 < argc)
            args.optimizer_type = argv[++i];
        else if(arg == "--momentum" && i + 1 < argc)
            args.momentum = static_cast<float>(std::atof(argv[++i]));
        else if(arg == "--weight-decay" && i + 1 < argc)
            args.weight_decay = static_cast<float>(std::atof(argv[++i]));
        else if(arg == "--beta1" && i + 1 < argc)
            args.beta1 = static_cast<float>(std::atof(argv[++i]));
        else if(arg == "--beta2" && i + 1 < argc)
            args.beta2 = static_cast<float>(std::atof(argv[++i]));
        else if(arg == "--save-model" && i + 1 < argc)
            args.save_model = argv[++i];
        else if(arg == "--load-model" && i + 1 < argc)
            args.load_model = argv[++i];
        else if(arg == "--save-optim" && i + 1 < argc)
            args.save_optim = argv[++i];
        else if(arg == "--load-optim" && i + 1 < argc)
            args.load_optim = argv[++i];
        else if(arg == "--save-config" && i + 1 < argc)
            args.save_config = argv[++i];
        else if(arg == "--load-config" && i + 1 < argc)
            args.load_config = argv[++i];
        else if(arg == "--export-hf" && i + 1 < argc)
            args.export_hf = argv[++i];
        else if(arg == "--import-hf" && i + 1 < argc)
            args.import_hf = argv[++i];
        else if(arg == "--export-hf-optim" && i + 1 < argc)
            args.export_hf_optim = argv[++i];
        else if(arg == "--import-hf-optim" && i + 1 < argc)
            args.import_hf_optim = argv[++i];
        else
        {
            std::cerr << "Unknown option: " << arg << "\n";
        }
    }
    return args;
}

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    const Index input_dim  = 128;
    const Index hidden_dim = 256;
    const Index output_dim = 10;
    const Index depth      = 5;
    const Index batch_size = 32;

    std::cout << "=== Deep ReLU Training Example ===\n"
              << "Architecture: " << input_dim << " -> "
              << hidden_dim << " (x" << (depth - 1) << " hidden) -> "
              << output_dim << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Training iterations: " << args.num_iters << "\n"
              << "Optimizer: " << args.optimizer_type << "\n"
              << "Learning rate: " << args.learning_rate << "\n\n";

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

    auto* residual = add(1.0, output, -1.0, target, "residual");

    Scalar loss_scale = 1.0 / static_cast<Scalar>(batch_size * output_dim);
    auto* loss = mse_loss(residual, "loss", loss_scale);
    loss->mark_output(true);

    auto [loss_grad, loss_grad_first] =
        graph.get_or_create_grad(loss, "loss_grad");
    graph::tensor::fill(Scalar(1.0), loss_grad->data());

    // ---- Backward pass ----
    loss->backward(true);

    // ---- Create optimizer ----
    std::unique_ptr<Optimizer> optimizer;
    if(args.optimizer_type == "sgd")
    {
        optimizer = std::make_unique<SGD>(
            &graph, &model,
            args.learning_rate,
            args.momentum,
            args.weight_decay);
    }
    else if(args.optimizer_type == "adam")
    {
        optimizer = std::make_unique<Adam>(
            &graph, &model,
            args.learning_rate,
            args.beta1,
            args.beta2,
            1e-8,
            args.weight_decay);
    }
    else if(args.optimizer_type == "adamw")
    {
        optimizer = std::make_unique<AdamW>(
            &graph, &model,
            args.learning_rate,
            args.beta1,
            args.beta2,
            1e-8,
            args.weight_decay);
    }
    else
    {
        std::cerr << "Unknown optimizer: " << args.optimizer_type << "\n";
        return 1;
    }

    // Load optimizer config (hyperparameters) if requested
    if(!args.load_config.empty())
    {
        optimizer->load_config(args.load_config);
        std::cout << "Loaded optimizer config from: "
                  << args.load_config << "\n";
    }

    // Add optimizer step ops to the graph
    optimizer->step();

    std::cout << "Model structure:\n" << model.to_string() << "\n";
    std::cout << "Optimizer: " << optimizer->repr() << "\n";

    auto params = model.named_parameters_recursive();
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

    std::vector<float> target_data(
        static_cast<std::size_t>(batch_size * output_dim));
    std::uniform_real_distribution<float> target_dist(-1.0f, 1.0f);
    for(auto& v : target_data) v = target_dist(gen);

    // Initialize weights (Kaiming uniform) unless loading from file
    if(!args.load_model.empty())
    {
        model.load(args.load_model);
        std::cout << "Loaded model weights from: " << args.load_model << "\n";
    }
    else if(!args.import_hf.empty())
    {
        io::SafeTensorsReader hf_reader(args.import_hf);
        model.import_hf(hf_reader, "");
        std::cout << "Imported model weights from HF: "
                  << args.import_hf << "\n";
    }
    else
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
            writer.add_tensor(name, tensor->dtype(), shape64,
                              std::move(bytes));
        }
        const std::string weights_path =
            "/tmp/deep_relu_training_weights.safetensors";
        writer.write(weights_path);
        model.load(weights_path);
    }

    // ---- Compile the graph ----
    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph.tensor_graph());

    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    // ---- Bind initial data ----
    runtime.bind_data("input", input_data);
    runtime.bind_data("target", target_data);

    // Load or zero-initialize optimizer state
    if(!args.load_optim.empty())
    {
        optimizer->load(args.load_optim);
        std::cout << "Loaded optimizer state from: "
                  << args.load_optim << "\n";
    }
    else if(!args.import_hf_optim.empty())
    {
        io::SafeTensorsReader hf_reader(args.import_hf_optim);
        optimizer->import_hf(hf_reader, "");
        std::cout << "Imported optimizer state from HF: "
                  << args.import_hf_optim << "\n";
    }
    else
    {
        auto state_tensors = optimizer->named_state_tensors();
        for(const auto& [sname, stensor] : state_tensors)
        {
            Index n = 1;
            for(auto d : stensor->shape()) n *= d;
            std::vector<float> zeros(static_cast<std::size_t>(n), 0.0f);
            runtime.bind_data(sname, zeros);
        }
    }

    // ---- Training loop ----
    std::cout << "Training...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    for(int iter = 0; iter < args.num_iters; ++iter)
    {
        runtime.execute();
        runtime.wait();

        auto loss_data = runtime.get_output<float>("loss");
        float loss_val = loss_data[0];

        if(iter == 0 || (iter + 1) % 10 == 0 || iter == args.num_iters - 1)
        {
            std::cout << "  Iter " << (iter + 1) << "/" << args.num_iters
                      << ": loss = " << loss_val << "\n";
        }

    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_start).count();

    std::cout << "\nTraining completed in " << total_us << " us";
    if(args.num_iters > 0)
    {
        std::cout << " (" << (total_us / args.num_iters) << " us/iter)";
    }
    std::cout << "\n";

    // ---- Sync trained data from runtime to bind_hints for saving ----
    bool need_model_sync = !args.save_model.empty()
                        || !args.export_hf.empty();
    bool need_optim_sync = !args.save_optim.empty()
                        || !args.export_hf_optim.empty();
    auto sync_tensor = [&runtime](NNGraph::TensorNode* t) {
        const auto& tname = t->name();
        std::vector<std::uint8_t> bytes;
        switch(t->dtype())
        {
            case DataType::FP64:
            {
                auto d = runtime.get_output<double>(tname);
                bytes.resize(d.size() * sizeof(double));
                std::memcpy(bytes.data(), d.data(), bytes.size());
                break;
            }
            case DataType::INT64:
            {
                auto d = runtime.get_output<std::int64_t>(tname);
                bytes.resize(d.size() * sizeof(std::int64_t));
                std::memcpy(bytes.data(), d.data(), bytes.size());
                break;
            }
            default:
            {
                auto d = runtime.get_output<float>(tname);
                bytes.resize(d.size() * sizeof(float));
                std::memcpy(bytes.data(), d.data(), bytes.size());
                break;
            }
        }
        t->data()->set_bind_hint(std::move(bytes));
    };
    if(need_model_sync)
    {
        for(const auto& [pname, ptensor] : params)
        {
            sync_tensor(ptensor);
        }
    }
    if(need_optim_sync)
    {
        optimizer->sync_from_runtime(runtime);
    }

    // ---- Save model weights ----
    if(!args.save_model.empty())
    {
        model.save(args.save_model);
        std::cout << "Saved model weights to: " << args.save_model << "\n";
    }

    // ---- Export model in HF format ----
    if(!args.export_hf.empty())
    {
        io::SafeTensorsWriter hf_writer;
        model.export_hf(hf_writer, "");
        hf_writer.write(args.export_hf);
        std::cout << "Exported model weights to HF: "
                  << args.export_hf << "\n";
    }

    // ---- Save optimizer state ----
    if(!args.save_optim.empty())
    {
        optimizer->save(args.save_optim);
        std::cout << "Saved optimizer state to: " << args.save_optim << "\n";
    }

    // ---- Export optimizer state in HF format ----
    if(!args.export_hf_optim.empty())
    {
        io::SafeTensorsWriter hf_writer;
        optimizer->export_hf(hf_writer, "");
        hf_writer.write(args.export_hf_optim);
        std::cout << "Exported optimizer state to HF: "
                  << args.export_hf_optim << "\n";
    }

    // ---- Save optimizer config (JSON) ----
    if(!args.save_config.empty())
    {
        optimizer->save_config(args.save_config);
        std::cout << "Saved optimizer config to: " << args.save_config << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
