/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/llama_generate.cc
 * Autoregressive text generation with a Llama causal LM.
 *
 * Workflow (two-step):
 *
 *   # Step 1 – Convert HF checkpoint and tokenize the prompt (Python):
 *   python examples/llama_generate.py \
 *       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
 *       --output-dir /tmp/nntile_llama \
 *       --prompt "The meaning of life is"
 *
 *   # Step 2 – Run autoregressive generation (this binary):
 *   ./llama_generate \
 *       --config /tmp/nntile_llama/config.json \
 *       --weights /tmp/nntile_llama/weights.safetensors \
 *       --prompt-ids "$(cat /tmp/nntile_llama/prompt_ids.txt)" \
 *       --max-tokens 32
 *
 * Alternatively, let the binary call the Python script automatically:
 *   ./llama_generate \
 *       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
 *       --prompt "The meaning of life is" \
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
#include <nntile/graph/kv_cache.hh>
#include <nntile/graph/model/llama/llama_causal.hh>
#include <nntile/graph/model/llama/llama_config.hh>
#include <nlohmann/json.hpp>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using json = nlohmann::json;

// ── Example constants ────────────────────────────────────────────────────

constexpr const char* DEFAULT_OUTPUT_DIR = "/tmp/nntile_llama";
constexpr int DEFAULT_MAX_TOKENS = 32;

// StarPU context: 1 CPU worker, 1 CUDA worker, OOC disabled
constexpr int CONTEXT_NUM_CPU = 1;
constexpr int CONTEXT_NUM_CUDA = 1;
constexpr int CONTEXT_OOC = 0;       // OOC disabled
constexpr int CONTEXT_OOC_SIZE = 16777216;  // 16 MiB OOC buffer (unused when OOC=0)
constexpr int CONTEXT_LOGGER = 0;    // logger disabled
constexpr int CONTEXT_VERBOSE = 0;
constexpr int CONTEXT_LOGGER_PORT = 5001;

// Llama config defaults (used when config.json omits fields)
constexpr int DEFAULT_VOCAB_SIZE = 32000;
constexpr int DEFAULT_HIDDEN_SIZE = 4096;
constexpr int DEFAULT_INTERMEDIATE_SIZE = 11008;
constexpr int DEFAULT_NUM_HIDDEN_LAYERS = 32;
constexpr int DEFAULT_NUM_ATTENTION_HEADS = 32;
constexpr int DEFAULT_NUM_KEY_VALUE_HEADS = 32;
constexpr float DEFAULT_RMS_NORM_EPS = 1e-6f;
constexpr int DEFAULT_EOS_TOKEN_ID = 2;
constexpr int DEFAULT_BOS_TOKEN_ID = 1;

constexpr int BATCH_SIZE = 1;  // single-sequence generation

constexpr int EXIT_OK = 0;
constexpr int EXIT_ERROR = 1;

// ── CLI helpers ──────────────────────────────────────────────────────────

struct Args
{
    std::string config_path;
    std::string weights_path;
    std::string prompt_ids_str;
    std::string model;
    std::string prompt;
    std::string output_dir = DEFAULT_OUTPUT_DIR;
    int max_tokens = DEFAULT_MAX_TOKENS;
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
        else if(arg == "--prompt-ids" && i + 1 < argc)
            a.prompt_ids_str = argv[++i];
        else if((arg == "--model" || arg == "--model-dir") && i + 1 < argc)
            a.model = argv[++i];
        else if(arg == "--prompt" && i + 1 < argc)
            a.prompt = argv[++i];
        else if(arg == "--output-dir" && i + 1 < argc)
            a.output_dir = argv[++i];
        else if(arg == "--max-tokens" && i + 1 < argc)
            a.max_tokens = std::stoi(argv[++i]);
        else if(arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n\n"
                << "Option A – provide pre-converted files:\n"
                << "  --config <path>       config.json (from llama_generate.py)\n"
                << "  --weights <path>      weights.safetensors\n"
                << "  --prompt-ids <ids>    comma-separated token IDs\n\n"
                << "Option B – let the binary call Python automatically:\n"
                << "  --model <name|path>   HF model name or local path\n"
                << "  --prompt <text>       text prompt\n"
                << "  --output-dir <path>   scratch directory "
                   "(default: " << DEFAULT_OUTPUT_DIR << ")\n\n"
                << "Common:\n"
                << "  --max-tokens <N>      tokens to generate (default: "
                   << DEFAULT_MAX_TOKENS << ")\n";
            std::exit(EXIT_OK);
        }
    }
    return a;
}

static std::vector<std::int64_t> parse_ids(const std::string& s)
{
    // If s looks like a file path, read the comma-separated IDs from it.
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

// Extract int from JSON; accepts both number and string (e.g. "32000").
static int config_get_int(const json& j, const char* key, int default_val)
{
    if(!j.contains(key))
        return default_val;
    const auto& v = j[key];
    if(v.is_number_integer())
        return v.get<int>();
    if(v.is_number_float())
        return static_cast<int>(v.get<double>());
    if(v.is_string())
        return std::stoi(v.get<std::string>());
    throw std::runtime_error(std::string("config: '") + key +
                            "' must be int or string, got " + v.type_name());
}

// Extract float from JSON; accepts both number and string.
static float config_get_float(const json& j, const char* key, float default_val)
{
    if(!j.contains(key))
        return default_val;
    const auto& v = j[key];
    if(v.is_number_integer() || v.is_number_float())
        return static_cast<float>(v.get<double>());
    if(v.is_string())
        return std::stof(v.get<std::string>());
    throw std::runtime_error(std::string("config: '") + key +
                            "' must be number or string, got " + v.type_name());
}

static LlamaConfig load_config(const std::string& path)
{
    std::ifstream f(path);
    if(!f.good())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }
    json j = json::parse(f);

    LlamaConfig cfg;
    cfg.vocab_size           = config_get_int(j, "vocab_size", DEFAULT_VOCAB_SIZE);
    cfg.hidden_size          = config_get_int(j, "hidden_size", DEFAULT_HIDDEN_SIZE);
    cfg.intermediate_size    = config_get_int(j, "intermediate_size", DEFAULT_INTERMEDIATE_SIZE);
    cfg.num_hidden_layers    = config_get_int(j, "num_hidden_layers", DEFAULT_NUM_HIDDEN_LAYERS);
    cfg.num_attention_heads = config_get_int(j, "num_attention_heads", DEFAULT_NUM_ATTENTION_HEADS);
    cfg.num_key_value_heads = config_get_int(j, "num_key_value_heads", DEFAULT_NUM_KEY_VALUE_HEADS);
    cfg.rms_norm_eps         = config_get_float(j, "rms_norm_eps", DEFAULT_RMS_NORM_EPS);
    cfg.eos_token_id         = config_get_int(j, "eos_token_id", DEFAULT_EOS_TOKEN_ID);
    cfg.bos_token_id         = config_get_int(j, "bos_token_id", DEFAULT_BOS_TOKEN_ID);
    cfg.compute_head_dim();
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
        "examples/llama_generate.py",
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
    };
    if(!args.prompt.empty())
    {
        argv_strs.push_back("--prompt");
        argv_strs.push_back(args.prompt);
    }

    std::vector<char*> argv;
    argv.reserve(argv_strs.size() + 1);
    for(auto& s : argv_strs)
        argv.push_back(s.data());
    argv.push_back(nullptr);

    std::cout << "Running: python3 examples/llama_generate.py --model "
              << args.model << " --output-dir " << args.output_dir;
    if(!args.prompt.empty())
        std::cout << " --prompt " << args.prompt;
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
        _exit(127);  // exec failed: conventional "command not found"
    }
    int status;
    if(waitpid(pid, &status, 0) != pid)  // 0 = no options (blocking wait)
    {
        std::cerr << "waitpid() failed\n";
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;  // 0 = success
}

// ── Read prompt_ids.txt ──────────────────────────────────────────────────

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
    // logits layout: (vocab, seq, batch=1) column-major
    // position s occupies offsets [s*vocab, (s+1)*vocab)
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

    // If --model was given, call the Python converter first.
    if(!args.model.empty())
    {
        if(!run_python_conversion(args))
        {
            std::cerr << "Python conversion failed.\n";
            return EXIT_ERROR;
        }
        args.config_path  = args.output_dir + "/config.json";
        args.weights_path = args.output_dir + "/weights.safetensors";

        if(args.prompt_ids_str.empty())
        {
            std::string ids_file = args.output_dir + "/prompt_ids.txt";
            args.prompt_ids_str = read_file_trimmed(ids_file);
        }
    }

    if(args.config_path.empty() || args.weights_path.empty()
       || args.prompt_ids_str.empty())
    {
        std::cerr
            << "Error: provide either (--model + --prompt) or "
               "(--config + --weights + --prompt-ids).\n"
               "Run with --help for usage.\n";
        return EXIT_ERROR;
    }

    // ── Load config ──────────────────────────────────────────────────
    LlamaConfig config = load_config(args.config_path);
    std::cout << "Config: hidden=" << config.hidden_size
              << "  layers=" << config.num_hidden_layers
              << "  heads=" << config.num_attention_heads
              << "  kv_heads=" << config.num_key_value_heads
              << "  vocab=" << config.vocab_size << "\n";

    // ── Parse prompt token IDs ───────────────────────────────────────
    std::vector<std::int64_t> tokens = parse_ids(args.prompt_ids_str);
    if(tokens.empty())
    {
        std::cerr << "Error: prompt token list is empty.\n";
        return EXIT_ERROR;
    }
    std::cout << "Prompt: " << tokens.size() << " tokens\n";

    // ── Cache all weights in memory ──────────────────────────────────
    std::cout << "Loading weights from " << args.weights_path << " ...\n";
    WeightCache weights = load_weights_to_memory(args.weights_path);
    std::cout << "Cached " << weights.size() << " parameter tensors\n";

    // ── StarPU context ───────────────────────────────────────────────
    Context context(CONTEXT_NUM_CPU, CONTEXT_NUM_CUDA, CONTEXT_OOC, "/tmp/nntile_ooc",
                    CONTEXT_OOC_SIZE, CONTEXT_LOGGER, "localhost", CONTEXT_LOGGER_PORT,
                    CONTEXT_VERBOSE);
    context.restrict_cuda();

    // ── KV cache setup ───────────────────────────────────────────────
    KVCache kv_cache(config.num_hidden_layers,
                     config.head_dim,
                     config.num_key_value_heads,
                     static_cast<Index>(tokens.size()) + args.max_tokens,
                     BATCH_SIZE);
    kv_cache.reset();

    // ── Autoregressive generation loop (with KV cache) ────────────────
    std::cout << "\n--- Generating (greedy, with KV-cache) ---\n";
    for(int step = 0; step < args.max_tokens; ++step)
    {
        const bool is_prefill = (step == 0);
        Index seq_len = is_prefill
            ? static_cast<Index>(tokens.size())
            : 1;  // decode: one token at a time

        // Build a fresh graph for the current step.
        NNGraph graph("llama_step");

        auto* input_ids = graph.tensor(
            {seq_len, BATCH_SIZE}, "input_ids", DataType::INT64, false);
        input_ids->mark_input(true);

        kv_cache.create_tensors(&graph, "kv_cache");

        LlamaCausal model(&graph, "model", config);
        auto* output = model.forward(input_ids, nullptr, nullptr, nullptr,
                                     &kv_cache);
        output->mark_output(true);

        apply_weight_cache(model, weights);

        TensorGraph::Runtime runtime(graph.tensor_graph());
        runtime.compile();

        if(is_prefill)
        {
            runtime.bind_data("input_ids", tokens);
        }
        else
        {
            runtime.bind_data("input_ids",
                std::vector<std::int64_t>{tokens.back()});
        }

        kv_cache.bind(runtime, "kv_cache");

        runtime.execute();
        runtime.wait();

        kv_cache.update_from(runtime, "kv_cache");

        auto logits = runtime.get_output<float>(output->name());

        std::int64_t next_id = argmax_last_position(
            logits, config.vocab_size, seq_len);
        tokens.push_back(next_id);
        kv_cache.advance(seq_len);

        std::cout << "step " << (step + 1) << "  token=" << next_id << "\n";

        if(next_id == config.eos_token_id)
        {
            std::cout << "EOS reached.\n";
            break;
        }
    }

    // ── Print the full generated sequence ────────────────────────────
    std::cout << "\nGenerated token IDs:";
    for(auto id : tokens)
    {
        std::cout << " " << id;
    }
    std::cout << "\n";

    // Hint: decode these IDs back to text with the tokenizer, e.g.:
    //   python3 -c "
    //   from transformers import AutoTokenizer
    //   t = AutoTokenizer.from_pretrained('/path/to/llama-hf')
    //   print(t.decode([...]))"
    std::cout << "\nTo decode, run:\n"
              << "  python3 -c \"from transformers import AutoTokenizer; "
              << "t = AutoTokenizer.from_pretrained('<model-dir>'); "
              << "print(t.decode([";
    for(std::size_t i = 0; i < tokens.size(); ++i)
    {
        if(i > 0) std::cout << ",";
        std::cout << tokens[i];
    }
    std::cout << "]))\"\n";

    return EXIT_OK;
}
