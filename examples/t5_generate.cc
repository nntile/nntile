/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/t5_generate.cc
 * Autoregressive text generation with T5 encoder-decoder.
 *
 * Workflow (two-step):
 *
 *   # Step 1 – Convert HF checkpoint and tokenize (Python):
 *   python examples/t5_generate.py \
 *       --model google-t5/t5-small \
 *       --output-dir /tmp/nntile_t5 \
 *       --encoder-prompt "translate English to German: The house is wonderful."
 *
 *   # Step 2 – Run autoregressive generation (this binary):
 *   ./t5_generate \
 *       --config /tmp/nntile_t5/config.json \
 *       --weights /tmp/nntile_t5/weights.safetensors \
 *       --encoder-ids "$(cat /tmp/nntile_t5/encoder_ids.txt)" \
 *       --decoder-ids "$(cat /tmp/nntile_t5/decoder_ids.txt)" \
 *       --max-tokens 32
 *
 * @version 1.1.0
 * */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <nntile.hh>
#include <nntile/graph/io/safetensors.hh>
#include <nntile/graph/model/t5/t5_for_conditional_generation.hh>
#include <nntile/graph/model/t5/t5_config.hh>
#include <nlohmann/json.hpp>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::t5;
using json = nlohmann::json;

// ── CLI helpers ──────────────────────────────────────────────────────────

struct Args
{
    std::string config_path;
    std::string weights_path;
    std::string encoder_ids_str;
    std::string decoder_ids_str;
    std::string model;
    std::string encoder_prompt;
    std::string output_dir = "/tmp/nntile_t5";
    int max_tokens = 32;
};

static Args parse_args(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--config" && i + 1 < argc)
            a.config_path = argv[++i];
        else if(arg == "--weights" && i + 1 < argc)
            a.weights_path = argv[++i];
        else if(arg == "--encoder-ids" && i + 1 < argc)
            a.encoder_ids_str = argv[++i];
        else if(arg == "--decoder-ids" && i + 1 < argc)
            a.decoder_ids_str = argv[++i];
        else if((arg == "--model" || arg == "--model-dir") && i + 1 < argc)
            a.model = argv[++i];
        else if(arg == "--encoder-prompt" && i + 1 < argc)
            a.encoder_prompt = argv[++i];
        else if(arg == "--output-dir" && i + 1 < argc)
            a.output_dir = argv[++i];
        else if(arg == "--max-tokens" && i + 1 < argc)
            a.max_tokens = std::stoi(argv[++i]);
        else if(arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n\n"
                << "Option A – provide pre-converted files:\n"
                << "  --config <path>       config.json (from t5_generate.py)\n"
                << "  --weights <path>      weights.safetensors\n"
                << "  --encoder-ids <ids>   comma-separated encoder token IDs\n"
                << "  --decoder-ids <ids>   comma-separated decoder start IDs\n\n"
                << "Option B – let the binary call Python automatically:\n"
                << "  --model <name|path>   HF model name or local path\n"
                << "  --encoder-prompt <text>  source text\n"
                << "  --output-dir <path>   scratch directory "
                   "(default: /tmp/nntile_t5)\n\n"
                << "Common:\n"
                << "  --max-tokens <N>      tokens to generate (default: 32)\n";
            std::exit(0);
        }
    }
    return a;
}

static std::vector<std::int64_t> parse_ids(const std::string& s)
{
    std::string data = s;
    {
        std::ifstream f(s);
        if(f.good())
        {
            std::string contents((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
            if(!contents.empty())
                data = contents;
        }
    }

    std::vector<std::int64_t> ids;
    std::istringstream ss(data);
    std::string tok;
    while(std::getline(ss, tok, ','))
    {
        while(!tok.empty() && (tok.back() == '\n' || tok.back() == '\r'))
            tok.pop_back();
        if(!tok.empty())
            ids.push_back(std::stoll(tok));
    }
    return ids;
}

// ── Config loader ────────────────────────────────────────────────────────

static T5Config load_config(const std::string& path)
{
    std::ifstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    json j = json::parse(f);

    T5Config cfg;
    cfg.vocab_size           = j.value("vocab_size", 32100);
    cfg.d_model              = j.value("d_model", 512);
    cfg.d_kv                 = j.value("d_kv", 64);
    cfg.d_ff                 = j.value("d_ff", 1024);
    cfg.num_layers           = j.value("num_layers", 6);
    cfg.num_decoder_layers   = j.value("num_decoder_layers", 6);
    cfg.num_heads            = j.value("num_heads", 8);
    cfg.layer_norm_epsilon   = j.value("layer_norm_epsilon", 1e-5f);
    cfg.eos_token_id         = j.value("eos_token_id", 1);
    cfg.pad_token_id         = j.value("pad_token_id", 0);
    cfg.decoder_start_token_id = j.value("decoder_start_token_id", 0);
    cfg.validate();
    return cfg;
}

// ── Weight cache ─────────────────────────────────────────────────────────

using WeightCache = std::map<std::string, std::vector<std::uint8_t>>;

static WeightCache load_weights_to_memory(const std::string& path)
{
    io::SafeTensorsReader reader(path);
    WeightCache cache;
    for(const auto& name : reader.tensor_names())
    {
        cache[name] = reader.read_tensor(name);
    }
    return cache;
}

static void apply_weight_cache(
    graph::module::Module& model,
    const WeightCache& cache)
{
    for(const auto& [name, tensor] : model.named_parameters_recursive())
    {
        auto it = cache.find(name);
        if(it != cache.end())
        {
            auto copy = it->second;
            tensor->data()->set_bind_hint(std::move(copy));
            tensor->mark_input(true);
        }
    }
}

// ── Python invocation ────────────────────────────────────────────────────

static bool run_python_conversion(const Args& args)
{
    std::vector<std::string> argv_strs = {
        "python3",
        "examples/t5_generate.py",
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
    };
    if(!args.encoder_prompt.empty())
    {
        argv_strs.push_back("--encoder-prompt");
        argv_strs.push_back(args.encoder_prompt);
    }

    std::vector<char*> argv;
    argv.reserve(argv_strs.size() + 1);
    for(auto& s : argv_strs)
        argv.push_back(s.data());
    argv.push_back(nullptr);

    std::cout << "Running: python3 examples/t5_generate.py --model "
              << args.model << " --output-dir " << args.output_dir;
    if(!args.encoder_prompt.empty())
        std::cout << " --encoder-prompt " << args.encoder_prompt;
    std::cout << "\n";

    pid_t pid = fork();
    if(pid < 0)
    {
        std::cerr << "fork() failed\n";
        return false;
    }
    if(pid == 0)
    {
        execvp("python3", argv.data());
        std::cerr << "execvp(python3) failed\n";
        _exit(127);
    }
    int status;
    if(waitpid(pid, &status, 0) != pid)
    {
        std::cerr << "waitpid() failed\n";
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static std::string read_file_trimmed(const std::string& path)
{
    std::ifstream f(path);
    if(!f.good())
        return {};
    std::string s((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    while(!s.empty() && (s.back() == '\n' || s.back() == '\r'))
        s.pop_back();
    return s;
}

// ── Greedy argmax over last-position logits ──────────────────────────────

static std::int64_t argmax_last_position(
    const std::vector<float>& logits,
    Index vocab_size,
    Index seq_len)
{
    auto offset = static_cast<std::size_t>((seq_len - 1) * vocab_size);
    auto begin = logits.begin()
        + static_cast<std::ptrdiff_t>(offset);
    auto end = begin + vocab_size;
    return static_cast<std::int64_t>(std::distance(
        begin, std::max_element(begin, end)));
}

// ── Main ─────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    if(!args.model.empty())
    {
        if(!run_python_conversion(args))
        {
            std::cerr << "Python conversion failed.\n";
            return 1;
        }
        args.config_path   = args.output_dir + "/config.json";
        args.weights_path  = args.output_dir + "/weights.safetensors";

        if(args.encoder_ids_str.empty())
        {
            args.encoder_ids_str = read_file_trimmed(
                args.output_dir + "/encoder_ids.txt");
        }
        if(args.decoder_ids_str.empty())
        {
            args.decoder_ids_str = read_file_trimmed(
                args.output_dir + "/decoder_ids.txt");
        }
    }

    if(args.config_path.empty() || args.weights_path.empty()
       || args.encoder_ids_str.empty() || args.decoder_ids_str.empty())
    {
        std::cerr
            << "Error: provide either (--model + --encoder-prompt) or "
               "(--config + --weights + --encoder-ids + --decoder-ids).\n"
               "Run with --help for usage.\n";
        return 1;
    }

    T5Config config = load_config(args.config_path);
    std::cout << "Config: d_model=" << config.d_model
              << "  encoder_layers=" << config.num_layers
              << "  decoder_layers=" << config.num_decoder_layers
              << "  heads=" << config.num_heads
              << "  vocab=" << config.vocab_size << "\n";

    std::vector<std::int64_t> encoder_tokens = parse_ids(args.encoder_ids_str);
    std::vector<std::int64_t> decoder_tokens = parse_ids(args.decoder_ids_str);

    if(encoder_tokens.empty())
    {
        std::cerr << "Error: encoder token list is empty.\n";
        return 1;
    }
    if(decoder_tokens.empty())
    {
        std::cerr << "Error: decoder token list is empty.\n";
        return 1;
    }
    std::cout << "Encoder: " << encoder_tokens.size() << " tokens\n";
    std::cout << "Decoder start: " << decoder_tokens.size() << " tokens\n";

    std::cout << "Loading weights from " << args.weights_path << " ...\n";
    WeightCache weights = load_weights_to_memory(args.weights_path);
    std::cout << "Cached " << weights.size() << " parameter tensors\n";

    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0,
                    "localhost", 5001, 0);

    std::cout << "\n--- Generating (greedy) ---\n";
    for(int step = 0; step < args.max_tokens; ++step)
    {
        Index enc_seq = static_cast<Index>(encoder_tokens.size());
        Index dec_seq = static_cast<Index>(decoder_tokens.size());

        NNGraph graph("t5_step");

        auto* encoder_ids = graph.tensor(
            {enc_seq, 1}, "encoder_input_ids", DataType::INT64, false);
        encoder_ids->mark_input(true);

        auto* decoder_ids = graph.tensor(
            {dec_seq, 1}, "decoder_input_ids", DataType::INT64, false);
        decoder_ids->mark_input(true);

        T5ForConditionalGeneration model(&graph, "model", config);
        auto* output = model.forward(encoder_ids, decoder_ids);
        output->mark_output(true);

        apply_weight_cache(model, weights);

        TensorGraph::Runtime runtime(graph.tensor_graph());
        runtime.compile();
        runtime.bind_data("encoder_input_ids", encoder_tokens);
        runtime.bind_data("decoder_input_ids", decoder_tokens);
        runtime.execute();
        runtime.wait();

        auto logits = runtime.get_output<float>(output->name());

        std::int64_t next_id = argmax_last_position(
            logits, config.vocab_size, dec_seq);
        decoder_tokens.push_back(next_id);

        std::cout << "step " << (step + 1) << "  token=" << next_id << "\n";

        if(next_id == config.eos_token_id)
        {
            std::cout << "EOS reached.\n";
            break;
        }
    }

    std::cout << "\nGenerated decoder token IDs:";
    for(auto id : decoder_tokens)
    {
        std::cout << " " << id;
    }
    std::cout << "\n";

    std::cout << "\nTo decode, run:\n"
              << "  python3 -c \"from transformers import AutoTokenizer; "
              << "t = AutoTokenizer.from_pretrained('<model-dir>'); "
              << "print(t.decode([";
    for(std::size_t i = 0; i < decoder_tokens.size(); ++i)
    {
        if(i > 0) std::cout << ",";
        std::cout << decoder_tokens[i];
    }
    std::cout << "]))\"\n";

    return 0;
}
