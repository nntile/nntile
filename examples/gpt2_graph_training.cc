/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/gpt2_graph_training.cc
 * GPT-2 causal LM training on the graph API.
 *
 * Data path: ``--train-bin`` is a flat ``uint16`` token stream read via mmap
 * (``TokenMemoryMap`` / ``CausalLmBatchIterator``). Each batch needs at least
 * ``(seq_len + 1) * batch`` ids; labels are the next-token targets for the
 * same windows. Optional ``--shuffle`` permutes sequence starts (``--seed``).
 *
 * Train.bin generation: use ``wrappers/python/examples/causal_lm_data_preparation.py``
 * with a GPT-2 tokenizer (e.g. ``--hf-tokenizer gpt2``). Point
 * ``--train-bin`` at the resulting ``train.bin``; ``Gpt2Config.vocab_size``
 * must cover token ids (``--tiny`` uses a 256-id toy vocab).
 *
 * Model path: ``Gpt2Causal`` forward takes ``input_ids``, ``position_ids``,
 * and a boolean causal attention mask; logits feed a scaled scalar cross-entropy
 * vs ``labels``. ``--tiny`` selects a built-in small config; ``--config`` loads
 * JSON ``Gpt2Config``; otherwise the tiny default applies.
 * ``--load-weights`` uses ``Module::load`` (SafeTensors).
 *
 * Training loop matches ``llama_graph_training``: clear grads, forward, loss,
 * ``backward(true)``, ``optimizer->step(scheduled_lr)``, ``finish_phase()``,
 * ``lower_and_compile()``, bind batch tensors, ``runtime.execute()``.
 *
 * Usage:
 *   ./gpt2_graph_training --train-bin /path/train.bin --seq 8 --batch 2
 *   ./gpt2_graph_training --train-bin data.bin --tiny --output-dir /tmp/out
 *
 * @version 1.1.0
 * */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "gpt2_config_json.hh"
#include <nntile.hh>
#include <nntile/graph/dataset/causal_lm_mmap.hh>
#include <nntile/graph/model/gpt2/gpt2_causal.hh>
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/nn/ops/sdpa_causal_mask.hh>
#include <nntile/graph/tensor/ops/clear.hh>
#include <random>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

using namespace nntile::core;
using namespace nntile::graph;
using namespace nntile::model::gpt2;
using namespace nntile::graph::dataset;
using namespace nntile::graph::optim;

namespace
{

constexpr int EXIT_OK = 0;
constexpr int EXIT_ERROR = 1;

constexpr int CONTEXT_NUM_CPU = 1;
constexpr int CONTEXT_NUM_CUDA = 0;
constexpr int CONTEXT_OOC = 0;
constexpr int CONTEXT_OOC_SIZE = 16777216;
constexpr int CONTEXT_LOGGER = 0;
constexpr int CONTEXT_VERBOSE = 0;
constexpr int CONTEXT_LOGGER_PORT = 5001;

struct Args
{
    std::string config_path;
    std::string load_weights;
    std::string output_dir;
    std::string train_bin;
    Index seq_len = 8;
    Index batch = 2;
    unsigned seed = 42;
    bool use_tiny = false;
    bool shuffle = false;
    std::size_t max_batches = 0;
    std::size_t epochs = 1;
    float learning_rate = 0.001f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int warmup_steps = 0;
};



using nntile::examples::load_gpt2_config_json;
using nntile::examples::save_gpt2_config_json;

static Gpt2Config make_tiny_config()
{
    Gpt2Config c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.max_position_embeddings = 512;
    c.layer_norm_eps = 1e-5f;
    c.validate();
    return c;
}

static bool take_string_opt(
    int argc, char **argv, int &i, char const *name, std::string &out)
{
    std::string const arg = argv[i];
    std::string const prefix = std::string(name) + "=";
    if (arg == name)
    {
        if (i + 1 >= argc)
        {
            return false;
        }
        out = argv[++i];
        return true;
    }
    if (arg.compare(0, prefix.size(), prefix) == 0)
    {
        out = arg.substr(prefix.size());
        return true;
    }
    return false;
}

static Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        std::string s;
        if (take_string_opt(argc, argv, i, "--config", a.config_path))
        {
            continue;
        }
        if (take_string_opt(argc, argv, i, "--load-weights", a.load_weights))
        {
            continue;
        }
        if (take_string_opt(argc, argv, i, "--output-dir", a.output_dir))
        {
            continue;
        }
        if (take_string_opt(argc, argv, i, "--train-bin", a.train_bin))
        {
            continue;
        }
        if (take_string_opt(argc, argv, i, "--max-batches", s))
        {
            a.max_batches = static_cast<std::size_t>(std::stoull(s));
            continue;
        }
        if (take_string_opt(argc, argv, i, "--epochs", s))
        {
            a.epochs = static_cast<std::size_t>(std::stoull(s));
            continue;
        }
        if (take_string_opt(argc, argv, i, "--lr", s))
        {
            a.learning_rate = std::stof(s);
            continue;
        }
        if (take_string_opt(argc, argv, i, "--weight-decay", s))
        {
            a.weight_decay = std::stof(s);
            continue;
        }
        if (take_string_opt(argc, argv, i, "--beta1", s))
        {
            a.beta1 = std::stof(s);
            continue;
        }
        if (take_string_opt(argc, argv, i, "--beta2", s))
        {
            a.beta2 = std::stof(s);
            continue;
        }
        if (take_string_opt(argc, argv, i, "--warmup-steps", s))
        {
            a.warmup_steps = std::stoi(s);
            continue;
        }
        if (take_string_opt(argc, argv, i, "--seq", s))
        {
            a.seq_len = static_cast<Index>(std::stoll(s));
            continue;
        }
        if (take_string_opt(argc, argv, i, "--batch", s))
        {
            a.batch = static_cast<Index>(std::stoll(s));
            continue;
        }
        if (take_string_opt(argc, argv, i, "--seed", s))
        {
            a.seed = static_cast<unsigned>(std::stoul(s));
            continue;
        }
        if (arg == "--tiny")
        {
            a.use_tiny = true;
            continue;
        }
        if (arg == "--shuffle")
        {
            a.shuffle = true;
            continue;
        }
        if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --config <path>       GPT-2 JSON (optional if --tiny)\n"
                << "  --tiny                use built-in tiny architecture\n"
                << "  --load-weights <path> SafeTensors checkpoint\n"
                << "  --output-dir <path>   write config.json + "
                   "model.safetensors\n"
                << "  --train-bin <path>    required: uint16 token file\n"
                << "  --shuffle             shuffle sequence starts\n"
                << "  --max-batches <N>     cap batches (0 = all)\n"
                << "  --epochs <N>          repeat mmap data (default 1)\n"
                << "  --lr <rate>           AdamW learning rate\n"
                << "  --weight-decay <v>    AdamW weight decay\n"
                << "  --beta1 / --beta2     AdamW betas\n"
                << "  --warmup-steps <N>    linear LR warmup\n"
                << "  --seq <N>             sequence length (default 8)\n"
                << "  --batch <N>           batch size (default 2)\n"
                << "  --seed <N>            RNG seed (default 42)\n"
                << "\n"
                << "Options may use --opt=value or --opt value.\n";
            std::exit(EXIT_OK);
        }
        if (arg == "--config" || arg == "--load-weights" ||
            arg == "--output-dir" || arg == "--train-bin" || arg == "--seq" ||
            arg == "--batch" || arg == "--seed" || arg == "--max-batches" ||
            arg == "--lr" || arg == "--weight-decay" || arg == "--beta1" ||
            arg == "--beta2" || arg == "--warmup-steps" || arg == "--epochs")
        {
            std::cerr << "gpt2_graph_training: " << arg
                      << " requires a value\n";
            std::exit(EXIT_ERROR);
        }
        if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "gpt2_graph_training: unknown option " << arg
                      << "\n";
            std::exit(EXIT_ERROR);
        }
        std::cerr << "gpt2_graph_training: unexpected argument " << arg
                  << "\n";
        std::exit(EXIT_ERROR);
    }
    return a;
}

static Scalar scheduled_lr(Index train_step, Args const &args)
{
    Scalar const peak = static_cast<Scalar>(args.learning_rate);
    if (args.warmup_steps <= 0)
    {
        return peak;
    }
    Index const w = static_cast<Index>(args.warmup_steps);
    if (train_step + 1 <= w)
    {
        return peak * static_cast<Scalar>(train_step + 1) /
               static_cast<Scalar>(w);
    }
    return peak;
}

static void fill_arange_position_ids(
    std::vector<std::int64_t> &pos, Index n_seq, Index n_batch)
{
    for (Index b = 0; b < n_batch; ++b)
    {
        for (Index s = 0; s < n_seq; ++s)
        {
            pos[s + n_seq * b] = static_cast<std::int64_t>(s);
        }
    }
}

static void init_random_parameter_hints(Gpt2Causal &model, std::mt19937 &gen)
{
    for (NNGraph::TensorNode *tensor : model.parameters_recursive())
    {
        const auto &shape = tensor->shape();
        Index nelems = 1;
        for (auto d : shape)
        {
            nelems *= d;
        }
        float fan_in = static_cast<float>(shape[0]);
        if (fan_in < 1.f)
        {
            fan_in = 1.f;
        }
        float limit = std::sqrt(1.0f / fan_in);
        std::uniform_real_distribution<float> wdist(-limit, limit);

        std::vector<float> data(static_cast<std::size_t>(nelems));
        for (auto &v : data)
        {
            v = wdist(gen);
        }
        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        tensor->data()->set_bind_hint(std::move(bytes));
    }
    model.mark_parameters_input_recursive();
}

static void sync_param_hint_from_runtime(
    Runtime &runtime, NNGraph::TensorNode *t)
{
    std::vector<std::uint8_t> bytes;
    switch (t->dtype())
    {
    case DataType::FP64:
    {
        auto d = runtime.get_output<double>(t);
        bytes.resize(d.size() * sizeof(double));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    case DataType::INT64:
    {
        auto d = runtime.get_output<std::int64_t>(t);
        bytes.resize(d.size() * sizeof(std::int64_t));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    default:
    {
        auto d = runtime.get_output<float>(t);
        bytes.resize(d.size() * sizeof(float));
        std::memcpy(bytes.data(), d.data(), bytes.size());
        break;
    }
    }
    t->data()->set_bind_hint(std::move(bytes));
}

} // namespace

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    if (args.train_bin.empty())
    {
        std::cerr << "gpt2_graph_training: --train-bin <path> is required\n";
        return EXIT_ERROR;
    }

    Gpt2Config config;
    if (args.use_tiny)
    {
        config = make_tiny_config();
    }
    else if (!args.config_path.empty())
    {
        config = load_gpt2_config_json(args.config_path);
    }
    else
    {
        config = make_tiny_config();
    }

    if (args.seq_len > config.max_position_embeddings)
    {
        std::cerr << "Warning: seq_len > max_position_embeddings.\n";
    }

    if (args.warmup_steps < 0)
    {
        std::cerr << "gpt2_graph_training: --warmup-steps must be >= 0\n";
        return EXIT_ERROR;
    }
    if (args.epochs == 0)
    {
        std::cerr << "gpt2_graph_training: --epochs must be >= 1\n";
        return EXIT_ERROR;
    }

    const Index n_seq = args.seq_len;
    const Index n_batch = args.batch;

    std::cout << "=== GPT-2 graph training (setup) ===\n"
              << "hidden=" << config.hidden_size
              << "  layers=" << config.num_hidden_layers
              << "  heads=" << config.num_attention_heads
              << "  seq=" << n_seq << "  batch=" << n_batch
              << "  epochs=" << args.epochs << "\n";

    Context context(CONTEXT_NUM_CPU,
        CONTEXT_NUM_CUDA,
        CONTEXT_OOC,
        "/tmp/nntile_ooc",
        CONTEXT_OOC_SIZE,
        CONTEXT_LOGGER,
        "localhost",
        CONTEXT_LOGGER_PORT,
        CONTEXT_VERBOSE);

    NNGraph graph("gpt2_graph_training");
    Gpt2Causal model(&graph, "model", config);

    auto *input_ids = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                          ->set_name("input_ids");
    auto *position_ids =
        graph.tensor({n_seq, n_batch}, DataType::INT64, false)
            ->set_name("position_ids");
    auto *attn_mask = graph.tensor({n_seq, n_seq}, DataType::BOOL, false)
                          ->set_name("attn_mask");
    input_ids->mark_input(true);
    position_ids->mark_input(true);
    attn_mask->mark_input(true);

    auto *labels = graph.tensor({n_seq, n_batch}, DataType::INT64, false)
                       ->set_name("labels");
    labels->mark_input(true);

    if (!args.load_weights.empty())
    {
        model.load(args.load_weights);
    }
    else
    {
        std::mt19937 gen(args.seed);
        init_random_parameter_hints(model, gen);
    }

    auto optimizer = std::make_unique<AdamW>(&graph,
        &model,
        static_cast<Scalar>(args.learning_rate),
        static_cast<Scalar>(args.beta1),
        static_cast<Scalar>(args.beta2),
        1e-8,
        static_cast<Scalar>(args.weight_decay));

    std::cout << "Optimizer: " << optimizer->repr() << "\n";
    if (args.warmup_steps > 0)
    {
        std::cout << "LR schedule: linear warmup " << args.warmup_steps
                  << " steps to lr=" << args.learning_rate << "\n";
    }

    std::vector<std::int64_t> pos_data(
        static_cast<std::size_t>(n_seq * n_batch));
    fill_arange_position_ids(pos_data, n_seq, n_batch);

    const std::size_t mask_n = static_cast<std::size_t>(n_seq * n_seq);
    std::vector<std::uint8_t> mask_data(mask_n);
    sdpa_causal_mask_bool_fortran_fill(n_seq, mask_data.data());

    graph.enable_auto_tensor_name_phase_suffix(true);

    const Scalar ce_scale = 1.0f / static_cast<Scalar>(n_seq * n_batch);

    bool bound_optimizer_state = false;

    TokenMemoryMap train_mmap(args.train_bin);
    CausalLmBatchConfig lcfg;
    lcfg.n_seq = n_seq;
    lcfg.n_batch = n_batch;
    lcfg.shuffle = args.shuffle;
    lcfg.seed = args.seed;
    CausalLmBatch mmap_batch;
    Index train_step = 0;

    for (std::size_t epoch = 0; epoch < args.epochs; ++epoch)
    {
        if (args.max_batches > 0 &&
            train_step >= static_cast<Index>(args.max_batches))
        {
            break;
        }
        CausalLmBatchIterator train_it(train_mmap, lcfg, config.vocab_size);
        for (;;)
        {
            if (!train_it.next(mmap_batch))
            {
                break;
            }
            if (args.max_batches > 0 &&
                train_step >= static_cast<Index>(args.max_batches))
            {
                break;
            }

            if (train_step > 0)
            {
                graph.reset_incremental_tile_state();
            }

            for (NNGraph::TensorNode *p : graph.parameters())
            {
                if (p->grad() != nullptr)
                {
                    nntile::graph::tensor::clear(p->grad()->data());
                }
            }

            NNGraph::TensorNode *logits =
                model.forward(input_ids, position_ids, attn_mask);
            if (logits == nullptr)
            {
                throw std::runtime_error(
                    "gpt2_graph_training: model.forward returned null");
            }

            std::string const loss_name =
                std::string("loss_s") + std::to_string(train_step);
            auto *loss = cross_entropy(logits, labels, 0, ce_scale, -100)
                             ->set_name(loss_name);
            loss->mark_output(true);

            std::string const loss_grad_name = loss_name + "_grad";
            auto [loss_grad, loss_grad_first] =
                graph.get_or_create_grad(loss, loss_grad_name);
            (void) loss_grad_first;
            nntile::graph::tensor::fill(Scalar(1.0), loss_grad->data());
            loss->backward(true);

            Scalar const step_lr = scheduled_lr(train_step, args);
            optimizer->step(step_lr);

            graph.finish_phase();
            graph.lower_and_compile();
            Runtime &runtime = graph.runtime();

            if (!bound_optimizer_state)
            {
                auto state_tensors = optimizer->named_state_tensors();
                for (const auto &[sname, stensor] : state_tensors)
                {
                    (void) sname;
                    Index n = 1;
                    for (auto d : stensor->shape())
                    {
                        n *= d;
                    }
                    std::vector<float> zeros(static_cast<std::size_t>(n), 0.0f);
                    runtime.bind_data(stensor, zeros);
                }
                bound_optimizer_state = true;
            }

            runtime.bind_data(input_ids, mmap_batch.input_ids);
            runtime.bind_data(labels, mmap_batch.target_ids);
            runtime.bind_data(position_ids, pos_data);
            runtime.bind_data(attn_mask, mask_data);

            auto t0 = std::chrono::high_resolution_clock::now();
            runtime.execute();
            runtime.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();

            float loss_val = runtime.get_output<float>(loss)[0];
            if (args.warmup_steps > 0)
            {
                std::cout << "Batch " << static_cast<long long>(train_step)
                          << "  lr=" << static_cast<float>(step_lr)
                          << "  loss=" << loss_val << "  (" << us << " us)\n";
            }
            else
            {
                std::cout << "Batch " << static_cast<long long>(train_step)
                          << "  loss=" << loss_val << "  (" << us << " us)\n";
            }

            for (NNGraph::TensorNode *ptensor : graph.parameters())
            {
                sync_param_hint_from_runtime(runtime, ptensor);
            }
            for (const auto &[sname, stensor] :
                optimizer->named_state_tensors())
            {
                (void) sname;
                sync_param_hint_from_runtime(runtime, stensor);
            }

            ++train_step;
        }
        if (args.max_batches > 0 &&
            train_step >= static_cast<Index>(args.max_batches))
        {
            break;
        }
    }

    if (train_step == 0)
    {
        std::cerr
            << "Warning: no full batches from --train-bin (need at least "
            << "((seq+1)*batch) uint16 tokens).\n";
    }
    else
    {
        std::cout << "Processed " << static_cast<long long>(train_step)
                  << " batch(es) from " << args.train_bin << "\n";
    }

    if (!args.output_dir.empty())
    {
        std::filesystem::create_directories(args.output_dir);
        const std::string cfg_path = args.output_dir + "/config.json";
        const std::string w_path = args.output_dir + "/model.safetensors";
        save_gpt2_config_json(config, cfg_path);
        if (graph.has_runtime())
        {
            for (NNGraph::TensorNode *ptensor : graph.parameters())
            {
                sync_param_hint_from_runtime(graph.runtime(), ptensor);
            }
        }
        model.save(w_path);
        std::cout << "Wrote " << cfg_path << "\n"
                  << "Wrote " << w_path << "\n";
    }

    return EXIT_OK;
}
