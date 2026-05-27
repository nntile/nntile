/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneox/gptneox_decoder.cc
 * Tests for GptneoxDecoder.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneox/gptneox_decoder.hh"

#include "context_fixture.hh"
#include "nntile/graph/nn/ops/add.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/gptneox/gptneox_config.hh"
#include "test_frobenius.hh"
#include "test_gptneox_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
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
    "GptneoxDecoder tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace decoder_fixture_stem
{

constexpr char gptneox_decoder[] = "gptneox_decoder";

} // namespace decoder_fixture_stem

namespace
{

using namespace nntile::graph::test::gptneox_fixture;

struct DecoderFixtureSpec
{
    GptneoxConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_decoder_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    DecoderFixtureSpec &out)
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
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
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
        if(G.contains("use_parallel_residual"))
        {
            out.config.use_parallel_residual =
                G.at("use_parallel_residual").get<bool>();
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

inline std::string decoder_fixture_safetensors_path(
    const std::string &data_dir, const DecoderFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, DecoderFixtureSpec &fx)
{
    const std::string dir = std::string(GPTNEOX_DATA_DIR);
    if (!try_load_decoder_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(decoder_fixture_safetensors_path(dir, fx));
    return st.good();
}

void decoder_forward_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string full_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("decoder_ref_") + fx.stem);
        auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);
        if(mask != nullptr)
        {
            fill_sdpa_causal_mask_bytes(fx.seq, mask_bytes);
        }

        GptneoxDecoder decoder(&g, "decoder", fx.config);
        decoder.load(full_path);

        auto *output = decoder.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
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

void decoder_backward_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string full_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        NNGraph g(std::string("decoder_bwd_") + fx.stem);
        auto *input =
            g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32, true)
                ->set_name("input");
        GptneoxRopeInputs rope;
        load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);
        if(mask != nullptr)
        {
            fill_sdpa_causal_mask_bytes(fx.seq, mask_bytes);
        }

        GptneoxDecoder decoder(&g, "decoder", fx.config);
        decoder.load(full_path);
        auto *output = decoder.forward(input, rope.sin, rope.cos, mask);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result = runtime.get_output<float>(input->grad());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}

void decoder_run_and_compare_ref(
    const DecoderFixtureSpec &fx,
    const char *ref_tensor,
    NNGraph &g,
    NNGraph::TensorNode *input,
    NNGraph::TensorNode *out,
    const GptneoxRopeInputs *rope,
    NNGraph::TensorNode *mask,
    const std::vector<std::uint8_t> *mask_bytes,
    const std::vector<float> &input_data)
{
    const std::string full_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);
    if(!reader.has_tensor(ref_tensor))
    {
        SKIP(std::string("Fixture missing intermediate tensor ") + ref_tensor);
    }

    input->mark_input(true);
    out->mark_output(true);
    if(rope != nullptr)
    {
        mark_rope_inputs(*rope);
    }
    mark_mask_input(mask);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(input, input_data);
    if(rope != nullptr)
    {
        bind_rope_inputs(runtime, *rope);
    }
    if(mask != nullptr && mask_bytes != nullptr)
    {
        bind_mask_input(runtime, mask, *mask_bytes);
    }
    runtime.execute();
    runtime.wait();
    std::vector<float> result = runtime.get_output<float>(out);

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor(ref_tensor);
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());
    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, fx.forward_tol);
}

void decoder_input_norm_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string full_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    NNGraph g("decoder_input_norm");
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    GptneoxDecoder decoder(&g, "decoder", fx.config);
    decoder.load(full_path);
    auto *out = decoder.input_norm().forward(input);
    decoder_run_and_compare_ref(
        fx, "input_norm_out", g, input, out, nullptr, nullptr, nullptr, input_data);
}

void decoder_mlp_out_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string full_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    NNGraph g("decoder_mlp");
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    GptneoxRopeInputs rope;
    load_gptneox_rope_inputs(g, reader, fx.config, fx.seq, fx.batch, rope);
    NNGraph::TensorNode *mask = nullptr;
    std::vector<std::uint8_t> mask_bytes;
    load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);
    if(mask != nullptr)
    {
        fill_sdpa_causal_mask_bytes(fx.seq, mask_bytes);
    }
    GptneoxDecoder decoder(&g, "decoder", fx.config);
    decoder.load(full_path);
    auto *x_norm = decoder.input_norm().forward(input);
    auto *attn_out =
        decoder.attention().forward(x_norm, rope.sin, rope.cos, mask);
    auto *post_attn = add(1.0, input, 1.0, attn_out);
    auto *mlp_in = fx.config.use_parallel_residual
        ? decoder.post_attn_norm().forward(input)
        : decoder.post_attn_norm().forward(post_attn);
    auto *out = decoder.mlp().forward(mlp_in);
    decoder_run_and_compare_ref(fx,
        "mlp_out",
        g,
        input,
        out,
        &rope,
        mask,
        &mask_bytes,
        input_data);
}

} // namespace

TEST_CASE("GptneoxDecoder forward builds output", "[model][gptneox]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("Missing or invalid gptneox_decoder.json / .safetensors.");
    }
    NNGraph g("gptneox_decoder");
    GptneoxDecoder decoder(&g, "decoder", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = decoder.forward(input, nullptr, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("GptneoxDecoder load from safetensors roundtrip", "[model][gptneox][io]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("Missing or invalid gptneox_decoder.json / .safetensors.");
    }
    const std::string data_path =
        decoder_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoxDecoder decoder1(&g1, "decoder", fx.config);
    decoder1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneox_decoder_roundtrip.safetensors";
    decoder1.save(save_path);

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
    "GptneoxDecoder input_norm matches PyTorch reference", "[model][gptneox]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("GPT-NeoX decoder fixture not found.");
    }
    decoder_input_norm_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxDecoder mlp matches PyTorch reference", "[model][gptneox]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("GPT-NeoX decoder fixture not found.");
    }
    decoder_mlp_out_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxDecoder forward matches PyTorch reference", "[model][gptneox]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("GPT-NeoX decoder fixture not found.");
    }
    decoder_forward_compare_ref(fx);
}


TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxDecoder backward matches PyTorch reference",
    "[model][gptneox]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::gptneox_decoder, fx))
    {
        SKIP("GPT-NeoX decoder fixture not found.");
    }
    decoder_backward_compare_ref(fx);
}

#endif
