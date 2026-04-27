/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_model.cc
 * Tests for LlamaModel.
 *
 * Each reference bundle is a **pair**: ``<stem>.json`` (``Llama`` fields for
 * ``LlamaConfig`` plus ``sequence_length``, ``batch``, tolerances) and
 * ``<stem>.safetensors`` (weights, ``input_ids``, ``rope_cos`` / ``rope_sin``
 * from ``LlamaModel.rotary_emb`` on embeddings as in HuggingFace,
 * ``attn_mask`` for causal ``sdpa_eager``, ``output_ref``). Tests load the
 * JSON, build ``LlamaModel``, ``load()`` weights, then bind RoPE and mask for
 * ``forward`` to match the PyTorch reference.
 * Pairs are produced by ``generate_test_data.py`` (``--block model`` /
 * ``model_gqa``).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/model/llama/llama_model.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

#ifndef LLAMA_DATA_DIR

TEST_CASE("LlamaModel tests skipped (LLAMA_DATA_DIR undefined)", "[model][llama]")
{
    SKIP("LLAMA_DATA_DIR not defined at compile time.");
}

#else

//! Basenames (no extension) for paired ``.json`` / ``.safetensors`` in
//! ``LLAMA_DATA_DIR`` — must match ``generate_test_data.py`` output.
namespace model_fixture_stem
{

constexpr char llama_model[] = "llama_model";
constexpr char llama_model_gqa[] = "llama_model_gqa";

} // namespace model_fixture_stem

namespace
{

//! Parsed ``<stem>.json`` (``version`` 2) next to ``<stem>.safetensors``.
struct ModelFixtureSpec
{
    LlamaConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline Index json_index(const nlohmann::json& o, const char* key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

inline bool try_load_model_fixture_spec(
    const std::string& data_dir,
    const char* stem_cstr,
    ModelFixtureSpec& out)
{
    out = {};
    out.stem = stem_cstr;
    const std::string jpath = data_dir + "/" + out.stem + ".json";
    std::ifstream jf(jpath);
    if(!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        if(j.at("version").get<int>() != 2)
        {
            return false;
        }
        if(j.at("stem").get<std::string>() != out.stem)
        {
            return false;
        }
        const std::string expected_st = out.stem + ".safetensors";
        if(j.at("safetensors").get<std::string>() != expected_st)
        {
            return false;
        }
        const auto& L = j.at("llama");
        out.config.vocab_size = json_index(L, "vocab_size");
        out.config.hidden_size = json_index(L, "hidden_size");
        out.config.intermediate_size = json_index(L, "intermediate_size");
        out.config.num_hidden_layers = json_index(L, "num_hidden_layers");
        out.config.num_attention_heads = json_index(L, "num_attention_heads");
        out.config.num_key_value_heads = json_index(L, "num_key_value_heads");
        out.config.compute_head_dim();
        out.hidden = out.config.hidden_size;
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol = static_cast<float>(
            j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
    }
    catch(...)
    {
        return false;
    }
    return true;
}

inline std::string model_fixture_safetensors_path(
    const std::string& data_dir,
    const ModelFixtureSpec& spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

//! SKIP helper: JSON + safetensors must both exist and JSON must parse.
inline bool skip_unless_fixture_ready(const char* stem, ModelFixtureSpec& fx)
{
    const std::string dir = std::string(LLAMA_DATA_DIR);
    if(!try_load_model_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(model_fixture_safetensors_path(dir, fx));
    return st.good();
}

struct LlamaRopeInputs
{
    NNGraph::TensorNode* sin = nullptr;
    NNGraph::TensorNode* cos = nullptr;
    std::vector<float> sin_data;
    std::vector<float> cos_data;
};

inline bool load_llama_rope_inputs(
    NNGraph& g,
    const SafeTensorsReader& reader,
    const LlamaConfig& config,
    Index n_seq,
    Index n_batch,
    LlamaRopeInputs& out)
{
    out = {};
    if(!reader.has_tensor("rope_sin") || !reader.has_tensor("rope_cos"))
    {
        return false;
    }
    const Index head_dim = config.head_dim;
    if(head_dim % 2 != 0)
    {
        return false;
    }
    const Index half = head_dim / 2;
    out.sin = g.tensor({half, n_seq, n_batch}, "rope_sin", DataType::FP32);
    out.cos = g.tensor({half, n_seq, n_batch}, "rope_cos", DataType::FP32);
    auto read_f = [&](const char* name, std::vector<float>& dst)
    {
        std::vector<std::uint8_t> b = reader.read_tensor(name);
        dst.resize(b.size() / sizeof(float));
        std::memcpy(dst.data(), b.data(), b.size());
    };
    read_f("rope_sin", out.sin_data);
    read_f("rope_cos", out.cos_data);
    return true;
}

inline void mark_rope_inputs(const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    rope.sin->mark_input(true);
    rope.cos->mark_input(true);
}

inline void bind_rope_inputs(
    TileGraph::Runtime& runtime, const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data("rope_sin", rope.sin_data);
    runtime.bind_data("rope_cos", rope.cos_data);
}

//! Causal ``attn_mask`` ``(seq, seq)`` for ``sdpa_eager`` (1 = keep logit).
inline bool load_attn_mask_bool(
    NNGraph& g,
    const SafeTensorsReader& reader,
    Index n_seq,
    NNGraph::TensorNode*& out_mask,
    std::vector<std::uint8_t>& mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if(!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto& info = reader.tensor_info("attn_mask");
    if(info.shape.size() != 2 || info.shape[0] != n_seq
        || info.shape[1] != n_seq)
    {
        throw std::runtime_error(
            "Llama model test: attn_mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    out_mask = g.tensor({n_seq, n_seq}, "attn_mask", DataType::BOOL, false);
    auto raw = reader.read_tensor("attn_mask");
    if(info.dtype == DataType::BOOL)
    {
        if(raw.size() != n_el)
        {
            throw std::runtime_error(
                "Llama model test: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if(info.dtype == DataType::FP32)
    {
        if(raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "Llama model test: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto* p = reinterpret_cast<const float*>(raw.data());
        for(size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] =
                (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                              : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "Llama model test: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(NNGraph::TensorNode* mask)
{
    if(mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(
    TileGraph::Runtime& runtime,
    NNGraph::TensorNode* mask,
    const std::vector<std::uint8_t>& mask_bytes)
{
    if(mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask->name(), mask_bytes);
}

void model_forward_compare_ref(const ModelFixtureSpec& fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path = model_fixture_safetensors_path(data_dir, fx);
    const float tol = fx.forward_tol;
    const LlamaConfig& config = fx.config;
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;
    const Index hidden = fx.hidden;

    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("model_ref_") + fx.stem;
        NNGraph g(gname);
        auto* input_ids = g.tensor({n_seq, n_batch}, "input_ids", DataType::INT64);
        LlamaRopeInputs rope;
        REQUIRE(load_llama_rope_inputs(
            g, reader, config, n_seq, n_batch, rope));
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, n_seq, mask, mask_bytes));

        LlamaModel model(&g, "model", config);
        auto* output =
            model.forward(input_ids, rope.sin, rope.cos, mask);
        input_ids->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        model.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

} // namespace

TEST_CASE("LlamaModel forward builds output", "[model][llama]")
{
    ModelFixtureSpec fx;
    if(!skip_unless_fixture_ready(model_fixture_stem::llama_model, fx))
    {
        SKIP("Missing or invalid llama_model.json / .safetensors.");
    }
    NNGraph g("llama_model");
    LlamaModel model(&g, "model", fx.config);
    auto* input_ids = g.tensor({fx.seq, fx.batch}, "input_ids", DataType::INT64);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape()
        == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("LlamaModel GQA forward builds output", "[model][llama][gqa]")
{
    ModelFixtureSpec fx;
    if(!skip_unless_fixture_ready(model_fixture_stem::llama_model_gqa, fx))
    {
        SKIP("Missing or invalid llama_model_gqa.json / .safetensors.");
    }
    NNGraph g("llama_model_gqa");
    LlamaModel model(&g, "model", fx.config);
    auto* input_ids = g.tensor({fx.seq, fx.batch}, "input_ids", DataType::INT64);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape()
        == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("LlamaModel load from safetensors roundtrip", "[model][llama][io]")
{
    ModelFixtureSpec fx;
    if(!skip_unless_fixture_ready(model_fixture_stem::llama_model, fx))
    {
        SKIP("Missing or invalid llama_model.json / .safetensors.");
    }
    const std::string data_path =
        model_fixture_safetensors_path(std::string(LLAMA_DATA_DIR), fx);

    NNGraph g1("load_graph");
    LlamaModel model1(&g1, "model", fx.config);
    model1.load(data_path);

    const std::string save_path = "/tmp/nntile_llama_model_roundtrip.safetensors";
    model1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for(const auto& name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        auto orig = reader.read_tensor(name);
        auto loaded = reader2.read_tensor(name);
        REQUIRE(orig.size() == loaded.size());
        REQUIRE(orig == loaded);
    }

    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaModel matches PyTorch reference", "[model][llama]")
{
    ModelFixtureSpec fx;
    if(!skip_unless_fixture_ready(model_fixture_stem::llama_model, fx))
    {
        SKIP("Llama model fixture pair not found.");
    }
    model_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaModel GQA matches PyTorch reference", "[model][llama][gqa]")
{
    ModelFixtureSpec fx;
    if(!skip_unless_fixture_ready(model_fixture_stem::llama_model_gqa, fx))
    {
        SKIP("Llama model GQA fixture pair not found.");
    }
    model_forward_compare_ref(fx);
}

#endif
