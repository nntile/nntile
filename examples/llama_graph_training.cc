/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/llama_graph_training.cc
 * Llama causal LM on the graph API: random batch data, RoPE, causal mask,
 * optional load/save of config (JSON) and weights (SafeTensors).
 *
 * Usage:
 *   ./llama_graph_training --output-dir /tmp/out --seq 8 --batch 2
 *   ./llama_graph_training --config cfg.json --load-weights w.safetensors \
 *       --output-dir /tmp/out
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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntile.hh>
#include <nntile/graph/model/llama/llama_causal.hh>
#include <nntile/graph/model/llama/llama_causal_mask.hh>
#include <nntile/graph/model/llama/llama_config.hh>
#include <nntile/graph/model/llama/llama_rope.hh>
#include <nlohmann/json.hpp>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using json = nlohmann::json;

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
    Index seq_len = 8;
    Index batch = 2;
    unsigned seed = 42;
    bool use_tiny = false;
};

static int config_get_int(const json& j, const char* key, int default_val)
{
    if(!j.contains(key))
    {
        return default_val;
    }
    const auto& v = j[key];
    if(v.is_number_integer())
    {
        return v.get<int>();
    }
    if(v.is_number_float())
    {
        return static_cast<int>(v.get<double>());
    }
    if(v.is_string())
    {
        return std::stoi(v.get<std::string>());
    }
    throw std::runtime_error(
        std::string("config: '") + key + "' must be int or string");
}

static float config_get_float(const json& j, const char* key, float def)
{
    if(!j.contains(key))
    {
        return def;
    }
    const auto& v = j[key];
    if(v.is_number_integer() || v.is_number_float())
    {
        return static_cast<float>(v.get<double>());
    }
    if(v.is_string())
    {
        return std::stof(v.get<std::string>());
    }
    throw std::runtime_error(
        std::string("config: '") + key + "' must be number or string");
}

static LlamaConfig load_llama_config_json(const std::string& path)
{
    std::ifstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    json j = json::parse(f);
    LlamaConfig cfg;
    cfg.vocab_size = config_get_int(j, "vocab_size", 32000);
    cfg.hidden_size = config_get_int(j, "hidden_size", 4096);
    cfg.intermediate_size =
        config_get_int(j, "intermediate_size", 11008);
    cfg.num_hidden_layers =
        config_get_int(j, "num_hidden_layers", 32);
    cfg.num_attention_heads =
        config_get_int(j, "num_attention_heads", 32);
    cfg.num_key_value_heads =
        config_get_int(j, "num_key_value_heads", 32);
    cfg.max_position_embeddings =
        config_get_int(j, "max_position_embeddings", 2048);
    cfg.rms_norm_eps = config_get_float(j, "rms_norm_eps", 1e-6f);
    cfg.rope_theta = config_get_float(j, "rope_theta", 10000.0f);
    cfg.eos_token_id = config_get_int(j, "eos_token_id", 2);
    cfg.bos_token_id = config_get_int(j, "bos_token_id", 1);
    cfg.compute_head_dim();
    cfg.validate();
    return cfg;
}

static void save_llama_config_json(
    LlamaConfig const& cfg,
    const std::string& path)
{
    json j;
    j["vocab_size"] = cfg.vocab_size;
    j["hidden_size"] = cfg.hidden_size;
    j["intermediate_size"] = cfg.intermediate_size;
    j["num_hidden_layers"] = cfg.num_hidden_layers;
    j["num_attention_heads"] = cfg.num_attention_heads;
    j["num_key_value_heads"] = cfg.num_key_value_heads;
    j["max_position_embeddings"] = cfg.max_position_embeddings;
    j["rms_norm_eps"] = cfg.rms_norm_eps;
    j["rope_theta"] = cfg.rope_theta;
    j["eos_token_id"] = cfg.eos_token_id;
    j["bos_token_id"] = cfg.bos_token_id;
    j["name"] = cfg.name;
    std::ofstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot write config: " + path);
    }
    f << j.dump(2) << "\n";
}

//! Small default architecture for examples (not a pretrained checkpoint).
static LlamaConfig make_tiny_config()
{
    LlamaConfig c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.max_position_embeddings = 512;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 10000.0f;
    c.compute_head_dim();
    c.validate();
    return c;
}

//! ``--name value`` or ``--name=value``; advances ``i`` on two-token form.
static bool take_string_opt(
    int argc,
    char** argv,
    int& i,
    char const* name,
    std::string& out)
{
    std::string const arg = argv[i];
    std::string const prefix = std::string(name) + "=";
    if(arg == name)
    {
        if(i + 1 >= argc)
        {
            return false;
        }
        out = argv[++i];
        return true;
    }
    if(arg.compare(0, prefix.size(), prefix) == 0)
    {
        out = arg.substr(prefix.size());
        return true;
    }
    return false;
}

static Args parse_args(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        std::string s;
        if(take_string_opt(argc, argv, i, "--config", a.config_path))
        {
            continue;
        }
        if(take_string_opt(argc, argv, i, "--load-weights", a.load_weights))
        {
            continue;
        }
        if(take_string_opt(argc, argv, i, "--output-dir", a.output_dir))
        {
            continue;
        }
        if(take_string_opt(argc, argv, i, "--seq", s))
        {
            a.seq_len = static_cast<Index>(std::stoll(s));
            continue;
        }
        if(take_string_opt(argc, argv, i, "--batch", s))
        {
            a.batch = static_cast<Index>(std::stoll(s));
            continue;
        }
        if(take_string_opt(argc, argv, i, "--seed", s))
        {
            a.seed = static_cast<unsigned>(std::stoul(s));
            continue;
        }
        if(arg == "--tiny")
        {
            a.use_tiny = true;
            continue;
        }
        if(arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --config <path>       Llama JSON (optional if --tiny)\n"
                << "  --tiny                use built-in tiny architecture\n"
                << "  --load-weights <path> SafeTensors checkpoint\n"
                << "  --output-dir <path>   write config.json + "
                   "model.safetensors\n"
                << "  --seq <N>             sequence length (default 8)\n"
                << "  --batch <N>           batch size (default 2)\n"
                << "  --seed <N>            RNG seed (default 42)\n"
                << "\n"
                << "Options may use --opt=value or --opt value.\n";
            std::exit(EXIT_OK);
        }
        if(arg == "--config" || arg == "--load-weights" || arg == "--output-dir"
            || arg == "--seq" || arg == "--batch" || arg == "--seed")
        {
            std::cerr << "llama_graph_training: " << arg
                      << " requires a value\n";
            std::exit(EXIT_ERROR);
        }
        if(!arg.empty() && arg[0] == '-')
        {
            std::cerr << "llama_graph_training: unknown option " << arg << "\n";
            std::exit(EXIT_ERROR);
        }
        std::cerr << "llama_graph_training: unexpected argument " << arg
                  << "\n";
        std::exit(EXIT_ERROR);
    }
    return a;
}

static void fill_random_input_ids(
    std::vector<std::int64_t>& ids,
    Index vocab_size,
    std::mt19937_64& gen)
{
    std::uniform_int_distribution<std::int64_t> dist(
        0, static_cast<std::int64_t>(vocab_size - 1));
    for(auto& v : ids)
    {
        v = dist(gen);
    }
}

//! Standard prefill positions: ``(s, b)`` → ``s`` (HF-style arange per row).
static void fill_arange_position_ids(
    std::vector<std::int64_t>& pos,
    Index n_seq,
    Index n_batch)
{
    for(Index b = 0; b < n_batch; ++b)
    {
        for(Index s = 0; s < n_seq; ++s)
        {
            pos[s + n_seq * b] = static_cast<std::int64_t>(s);
        }
    }
}

static void init_random_parameter_hints(
    LlamaCausal& model,
    std::mt19937& gen)
{
    auto params = model.named_parameters_recursive();
    for(const auto& [name, tensor] : params)
    {
        const auto& shape = tensor->shape();
        Index nelems = 1;
        for(auto d : shape)
        {
            nelems *= d;
        }
        float fan_in = static_cast<float>(shape[0]);
        if(fan_in < 1.f)
        {
            fan_in = 1.f;
        }
        float limit = std::sqrt(1.0f / fan_in);
        std::uniform_real_distribution<float> wdist(-limit, limit);

        std::vector<float> data(static_cast<std::size_t>(nelems));
        for(auto& v : data)
        {
            v = wdist(gen);
        }
        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        tensor->data()->set_bind_hint(std::move(bytes));
    }
}

} // namespace

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    LlamaConfig config;
    if(args.use_tiny)
    {
        config = make_tiny_config();
    }
    else if(!args.config_path.empty())
    {
        config = load_llama_config_json(args.config_path);
    }
    else
    {
        config = make_tiny_config();
    }

    if(args.seq_len > config.max_position_embeddings)
    {
        std::cerr << "Warning: seq_len > max_position_embeddings.\n";
    }

    const Index n_seq = args.seq_len;
    const Index n_batch = args.batch;
    const Index half = config.head_dim / 2;

    std::cout << "=== Llama graph training (setup) ===\n"
              << "hidden=" << config.hidden_size
              << "  layers=" << config.num_hidden_layers
              << "  heads=" << config.num_attention_heads
              << "  kv_heads=" << config.num_key_value_heads
              << "  seq=" << n_seq << "  batch=" << n_batch << "\n";

    Context context(
        CONTEXT_NUM_CPU,
        CONTEXT_NUM_CUDA,
        CONTEXT_OOC,
        "/tmp/nntile_ooc",
        CONTEXT_OOC_SIZE,
        CONTEXT_LOGGER,
        "localhost",
        CONTEXT_LOGGER_PORT,
        CONTEXT_VERBOSE);

    NNGraph graph("llama_graph_training");
    LlamaCausal model(&graph, "model", config);

    auto* input_ids = graph.tensor(
        {n_seq, n_batch},
        "input_ids",
        DataType::INT64,
        false);
    auto* rope_sin = graph.tensor(
        {half, n_seq, n_batch},
        "rope_sin",
        DataType::FP32,
        false);
    auto* rope_cos = graph.tensor(
        {half, n_seq, n_batch},
        "rope_cos",
        DataType::FP32,
        false);
    auto* attn_mask = graph.tensor(
        {n_seq, n_seq},
        "attn_mask",
        DataType::BOOL,
        false);

    model.forward(
        input_ids,
        rope_sin,
        rope_cos,
        attn_mask,
        nullptr);

    NNGraph::TensorNode* logits = graph.get_tensor("model_logits");
    if(logits == nullptr)
    {
        throw std::runtime_error(
            "llama_graph_training: missing tensor model_logits");
    }
    logits->mark_output(true);

    input_ids->mark_input(true);
    rope_sin->mark_input(true);
    rope_cos->mark_input(true);
    attn_mask->mark_input(true);

    if(!args.load_weights.empty())
    {
        model.load(args.load_weights);
    }
    else
    {
        std::mt19937 gen(args.seed);
        init_random_parameter_hints(model, gen);
    }

    for(const auto& [pname, ptensor] : model.named_parameters_recursive())
    {
        (void)pname;
        ptensor->mark_input(true);
    }

    std::mt19937_64 gen64(args.seed);
    std::vector<std::int64_t> input_data(
        static_cast<std::size_t>(n_seq * n_batch));
    fill_random_input_ids(input_data, config.vocab_size, gen64);

    std::vector<std::int64_t> pos_data(
        static_cast<std::size_t>(n_seq * n_batch));
    fill_arange_position_ids(pos_data, n_seq, n_batch);

    const std::size_t rope_n =
        static_cast<std::size_t>(half * n_seq * n_batch);
    std::vector<float> sin_data(rope_n);
    std::vector<float> cos_data(rope_n);
    rope_sin_cos_from_position_ids(
        config,
        pos_data.data(),
        n_seq,
        n_batch,
        sin_data.data(),
        cos_data.data());

    const std::size_t mask_n =
        static_cast<std::size_t>(n_seq * n_seq);
    std::vector<std::uint8_t> mask_data(mask_n);
    sdpa_causal_mask_bool_fortran_fill(
        n_seq,
        mask_data.data());

    TileGraph tile_graph = TileGraph::from_tensor_graph(
        graph.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    runtime.bind_data("input_ids", input_data);
    runtime.bind_data("rope_sin", sin_data);
    runtime.bind_data("rope_cos", cos_data);
    runtime.bind_data("attn_mask", mask_data);

    auto t0 = std::chrono::high_resolution_clock::now();
    runtime.execute();
    runtime.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();

    auto logits_host =
        runtime.get_output<float>("model_logits");
    float sum = 0.f;
    for(float v : logits_host)
    {
        sum += v;
    }
    std::cout << "Forward ok (" << us << " us). Logits checksum (sum): "
              << sum << "\n";

    if(!args.output_dir.empty())
    {
        std::filesystem::create_directories(args.output_dir);
        const std::string cfg_path =
            args.output_dir + "/config.json";
        const std::string w_path =
            args.output_dir + "/model.safetensors";
        save_llama_config_json(config, cfg_path);
        model.save(w_path);
        std::cout << "Wrote " << cfg_path << "\n"
                  << "Wrote " << w_path << "\n";
    }

    return EXIT_OK;
}
