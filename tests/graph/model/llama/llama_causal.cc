/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_causal.cc
 * Tests for LlamaCausal.
 *
 * Each reference bundle is a **pair**: ``<stem>.json`` (same schema as
 * ``llama_model*.json`` — ``Llama`` fields, ``sequence_length``, ``batch``,
 * tolerances) and ``<stem>.safetensors`` (``model.model.*`` weights,
 * ``model.lm_head.weight``, ``input_ids``, ``rope_cos`` / ``rope_sin``,
 * ``attn_mask``, ``output_ref`` logits). Tests load the JSON, build
 * ``LlamaCausal``, ``load()`` weights, then bind RoPE and mask for
 * ``forward``. Pairs are produced by ``generate_test_data.py`` (``--block
 * causal`` /
 * ``causal_gqa``).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_causal.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "test_frobenius.hh"
#include "test_llama_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

#ifndef LLAMA_DATA_DIR

TEST_CASE(
    "LlamaCausal tests skipped (LLAMA_DATA_DIR undefined)", "[model][llama]")
{
    SKIP("LLAMA_DATA_DIR not defined at compile time.");
}

#else

namespace causal_fixture_stem
{

constexpr char llama_causal[] = "llama_causal";
constexpr char llama_causal_gqa[] = "llama_causal_gqa";

} // namespace causal_fixture_stem

namespace
{

using namespace nntile::test::llama_fixture;

//! Parsed ``<stem>.json`` (``version`` 2) — same fields as
//! ``llama_model*.json``.
struct CausalFixtureSpec
{
    LlamaConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_causal_fixture_spec(
    const std::string &data_dir, const char *stem_cstr, CausalFixtureSpec &out)
{
    out = {};
    out.stem = stem_cstr;
    const std::string jpath = data_dir + "/" + out.stem + ".json";
    std::ifstream jf(jpath);
    if (!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        if (j.at("version").get<int>() != 2)
        {
            return false;
        }
        if (j.at("stem").get<std::string>() != out.stem)
        {
            return false;
        }
        const std::string expected_st = out.stem + ".safetensors";
        if (j.at("safetensors").get<std::string>() != expected_st)
        {
            return false;
        }
        const auto &L = j.at("llama");
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
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline std::string causal_fixture_safetensors_path(
    const std::string &data_dir, const CausalFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, CausalFixtureSpec &fx)
{
    const std::string dir = std::string(LLAMA_DATA_DIR);
    if (!try_load_causal_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(causal_fixture_safetensors_path(dir, fx));
    return st.good();
}

void causal_forward_compare_ref(const CausalFixtureSpec &fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path =
        causal_fixture_safetensors_path(data_dir, fx);
    const float tol = fx.forward_tol;
    const LlamaConfig &config = fx.config;
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;

    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(
        ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("causal_ref_") + fx.stem;
        NNGraph g(gname);
        auto *input_ids =
            g.tensor({n_seq, n_batch}, DataType::INT64)->set_name("input_ids");
        LlamaRopeInputs rope;
        REQUIRE(
            load_llama_rope_inputs(g, reader, config, n_seq, n_batch, rope));
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, n_seq, mask, mask_bytes));

        LlamaCausal model(&g, "model", config);
        auto *output = model.forward(input_ids, rope.sin, rope.cos, mask);
        input_ids->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        model.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input_ids, ids_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

} // namespace

TEST_CASE("LlamaCausal forward builds output", "[model][llama]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::llama_causal, fx))
    {
        SKIP("Missing or invalid llama_causal.json / .safetensors.");
    }
    NNGraph g("llama_causal");
    LlamaCausal model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
    auto *output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.config.vocab_size, fx.seq, fx.batch}));
}

TEST_CASE("LlamaCausal GQA forward builds output", "[model][llama][gqa]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::llama_causal_gqa, fx))
    {
        SKIP("Missing or invalid llama_causal_gqa.json / .safetensors.");
    }
    NNGraph g("llama_causal_gqa");
    LlamaCausal model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
    auto *output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.config.vocab_size, fx.seq, fx.batch}));
}

TEST_CASE("LlamaCausal load from safetensors roundtrip", "[model][llama][io]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::llama_causal, fx))
    {
        SKIP("Missing or invalid llama_causal.json / .safetensors.");
    }
    const std::string data_path =
        causal_fixture_safetensors_path(std::string(LLAMA_DATA_DIR), fx);

    NNGraph g1("load_graph");
    LlamaCausal model1(&g1, "model", fx.config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_causal_roundtrip.safetensors";
    model1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for (const auto &name : reader2.tensor_names())
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
    "LlamaCausal matches PyTorch reference",
    "[model][llama]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::llama_causal, fx))
    {
        SKIP("Llama causal fixture pair not found.");
    }
    causal_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaCausal GQA matches PyTorch reference",
    "[model][llama][gqa]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::llama_causal_gqa, fx))
    {
        SKIP("Llama causal GQA fixture pair not found.");
    }
    causal_forward_compare_ref(fx);
}

#endif
