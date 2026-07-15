/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/gptneox/gptneox_causal.cc
 * Tests for GptneoxCausal.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_causal.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/gptneox/gptneox_config.hh"
#include "test_frobenius.hh"
#include "test_gptneox_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile;
using namespace nntile::model::gptneox;
using namespace nntile::io;

#ifndef GPTNEOX_DATA_DIR

TEST_CASE(
    "GptneoxCausal tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace causal_fixture_stem
{

constexpr char gptneox_causal[] = "gptneox_causal";

} // namespace causal_fixture_stem

namespace
{

using namespace nntile::test::gptneox_fixture;

struct CausalFixtureSpec
{
    GptneoxConfig config{};
    Index seq = 0;
    Index batch = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_causal_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    CausalFixtureSpec &out)
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
        const auto &G = j.at("gptneox");
        out.config.vocab_size = json_index(G, "vocab_size");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
        out.config.num_hidden_layers = json_index(G, "num_hidden_layers");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.head_dim = json_index(G, "head_dim");
        out.config.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        if(G.contains("rotary_pct"))
        {
            out.config.rotary_pct =
                static_cast<float>(G.at("rotary_pct").get<double>());
        }
        if(G.contains("rotary_emb_base"))
        {
            out.config.rotary_emb_base =
                static_cast<float>(G.at("rotary_emb_base").get<double>());
        }
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
        out.config.validate();
    }
    catch (...)
    {
        return false;
    }
    prepare_gptneox_config(out.config);
    return true;
}

inline std::string causal_fixture_safetensors_path(
    const std::string &data_dir, const CausalFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, CausalFixtureSpec &fx)
{
    const std::string dir = std::string(GPTNEOX_DATA_DIR);
    if (!try_load_causal_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(causal_fixture_safetensors_path(dir, fx));
    return st.good();
}

void causal_backward_compare_ref(const CausalFixtureSpec &fx)
{
    const std::string full_path =
        causal_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<std::uint8_t> ref_embed_bytes =
        reader.read_tensor("grad_embed_tokens_vocab");
    std::vector<float> grad_embed_ref(ref_embed_bytes.size() / sizeof(float));
    std::memcpy(
        grad_embed_ref.data(), ref_embed_bytes.data(), ref_embed_bytes.size());

    std::vector<float> grad_embed_result;
    {
        NNGraph g("causal_bwd");
        auto *input_ids =
            g.tensor({fx.batch, fx.seq}, DataType::INT64, true)
                ->set_name("input_ids");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoxCausal causal(&g, "model", fx.config);
        causal.load(full_path);
        auto *output = causal.forward(input_ids, rope.sin, rope.cos, mask);

        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        output->backward();

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input_ids, ids_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_embed_result = runtime.get_output<float>(
            causal.model()->embed_vocab_tensor()->grad());
    }

    REQUIRE(grad_embed_result.size() == grad_embed_ref.size());
    require_relative_frobenius_error(
        grad_embed_result, grad_embed_ref, fx.backward_tol);
}

} // namespace

TEST_CASE("GptneoxCausal forward builds output", "[model][gptneox]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneox_causal, fx))
    {
        SKIP("Missing or invalid gptneox_causal.json / .safetensors.");
    }
    NNGraph g("gptneox_causal");
    GptneoxCausal model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.batch, fx.seq}, DataType::INT64)->set_name("input_ids");
    auto *output = model.forward(input_ids, nullptr, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.batch, fx.seq, fx.config.vocab_size}));
}

TEST_CASE("GptneoxCausal load from safetensors roundtrip", "[model][gptneox][io]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneox_causal, fx))
    {
        SKIP("Missing or invalid gptneox_causal.json / .safetensors.");
    }
    const std::string data_path =
        causal_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoxCausal model1(&g1, "model", fx.config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneox_causal_roundtrip.safetensors";
    model1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);
    for (const auto &name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        REQUIRE(reader.read_tensor(name) == reader2.read_tensor(name));
    }
    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxCausal forward matches PyTorch reference", "[model][gptneox]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneox_causal, fx))
    {
        SKIP("GPT-NeoX causal fixture not found.");
    }
    const std::string full_path =
        causal_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("causal_ref");
        auto *input_ids =
            g.tensor({fx.batch, fx.seq}, DataType::INT64)->set_name("input_ids");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoxCausal causal(&g, "model", fx.config);
        causal.load(full_path);

        auto *output = causal.forward(input_ids, rope.sin, rope.cos, mask);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input_ids, ids_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, fx.forward_tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxCausal backward matches PyTorch reference",
    "[model][gptneox]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneox_causal, fx))
    {
        SKIP("GPT-NeoX causal fixture not found.");
    }
    causal_backward_compare_ref(fx);
}

#endif
