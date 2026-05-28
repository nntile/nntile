/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneox/gptneox_model.cc
 * Tests for GptneoxModel.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneox/gptneox_model.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/gptneox/gptneox_config.hh"
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

using namespace nntile::core;
using namespace nntile::graph;
using namespace nntile::graph::model::gptneox;
using namespace nntile::graph::io;

#ifndef GPTNEOX_DATA_DIR

TEST_CASE(
    "GptneoxModel tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace model_fixture_stem
{

constexpr char gptneox_model[] = "gptneox_model";

} // namespace model_fixture_stem

namespace
{

using namespace nntile::graph::test::gptneox_fixture;

struct ModelFixtureSpec
{
    GptneoxConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_model_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    ModelFixtureSpec &out)
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
        out.hidden = out.config.hidden_size;
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

inline std::string model_fixture_safetensors_path(
    const std::string &data_dir, const ModelFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, ModelFixtureSpec &fx)
{
    const std::string dir = std::string(GPTNEOX_DATA_DIR);
    if (!try_load_model_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(model_fixture_safetensors_path(dir, fx));
    return st.good();
}

void model_backward_compare_ref(const ModelFixtureSpec &fx)
{
    const std::string full_path =
        model_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
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
        NNGraph g("model_bwd");
        auto *input_ids =
            g.tensor({fx.seq, fx.batch}, DataType::INT64, true)
                ->set_name("input_ids");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoxModel model(&g, "model", fx.config);
        model.load(full_path);
        auto *output = model.forward(input_ids, rope.sin, rope.cos, mask);

        input_ids->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        model.embed_vocab_tensor()->grad()->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input_ids, ids_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_embed_result =
            runtime.get_output<float>(model.embed_vocab_tensor()->grad());
    }

    REQUIRE(grad_embed_result.size() == grad_embed_ref.size());
    require_relative_frobenius_error(
        grad_embed_result, grad_embed_ref, fx.backward_tol);
}


} // namespace

TEST_CASE("GptneoxModel forward builds output", "[model][gptneox]")
{
    ModelFixtureSpec fx;
    if (!skip_unless_fixture_ready(model_fixture_stem::gptneox_model, fx))
    {
        SKIP("Missing or invalid gptneox_model.json / .safetensors.");
    }
    NNGraph g("gptneox_model");
    GptneoxModel model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
    auto *output = model.forward(input_ids, nullptr, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("GptneoxModel load from safetensors roundtrip", "[model][gptneox][io]")
{
    ModelFixtureSpec fx;
    if (!skip_unless_fixture_ready(model_fixture_stem::gptneox_model, fx))
    {
        SKIP("Missing or invalid gptneox_model.json / .safetensors.");
    }
    const std::string data_path =
        model_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoxModel model1(&g1, "model", fx.config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneox_model_roundtrip.safetensors";
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

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxModel forward matches PyTorch reference", "[model][gptneox]")
{
    ModelFixtureSpec fx;
    if (!skip_unless_fixture_ready(model_fixture_stem::gptneox_model, fx))
    {
        SKIP("GPT-NeoX model fixture not found.");
    }
    const std::string full_path =
        model_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("model_ref");
        auto *input_ids =
            g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoxModel model(&g, "model", fx.config);
        model.load(full_path);

        auto *output = model.forward(input_ids, rope.sin, rope.cos, mask);
        input_ids->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
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


TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxModel backward matches PyTorch reference",
    "[model][gptneox]")
{
    ModelFixtureSpec fx;
    if (!skip_unless_fixture_ready(model_fixture_stem::gptneox_model, fx))
    {
        SKIP("GPT-NeoX model fixture not found.");
    }
    model_backward_compare_ref(fx);
}

#endif
