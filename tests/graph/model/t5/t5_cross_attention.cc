/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/t5/t5_cross_attention.cc
 * Tests for T5Attention (cross-attention).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_attention.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/t5/t5_config.hh"
#include "test_frobenius.hh"
#include "test_t5_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::t5;
using namespace nntile::graph::io;

#ifndef T5_DATA_DIR

TEST_CASE(
    "T5Attention cross tests skipped (T5_DATA_DIR undefined)", "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace cross_attn_fixture_stem
{

constexpr char t5_cross_attention[] = "t5_cross_attention";

} // namespace cross_attn_fixture_stem

namespace
{

using namespace nntile::test::t5_fixture;

struct CrossAttentionFixtureSpec
{
    T5Config config{};
    Index enc_seq = 0;
    Index dec_seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_cross_attention_fixture_spec(
    const std::string &data_dir,
    const char *stem_cstr,
    CrossAttentionFixtureSpec &out)
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
        const auto &T = j.at("t5");
        out.config.vocab_size = json_index(T, "vocab_size");
        out.config.d_model = json_index(T, "d_model");
        out.config.d_kv = json_index(T, "d_kv");
        out.config.d_ff = json_index(T, "d_ff");
        out.config.num_heads = json_index(T, "num_heads");
        out.config.num_layers = json_index(T, "num_layers");
        out.config.num_decoder_layers = json_index(T, "num_decoder_layers");
        out.config.layer_norm_epsilon = static_cast<float>(
            T.at("layer_norm_epsilon").get<double>());
        out.hidden = out.config.d_model;
        out.enc_seq = json_index(j, "encoder_sequence_length");
        out.dec_seq = json_index(j, "decoder_sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
        prepare_t5_config(out.config);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline std::string cross_attention_fixture_safetensors_path(
    const std::string &data_dir, const CrossAttentionFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(
    const char *stem, CrossAttentionFixtureSpec &fx)
{
    const std::string dir = std::string(T5_DATA_DIR);
    if (!try_load_cross_attention_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(cross_attention_fixture_safetensors_path(dir, fx));
    return st.good();
}

void cross_attention_forward_compare_ref(const CrossAttentionFixtureSpec &fx)
{
    const std::string full_path =
        cross_attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> enc_bytes =
        reader.read_tensor("encoder_input");
    std::vector<float> enc_data(enc_bytes.size() / sizeof(float));
    std::memcpy(enc_data.data(), enc_bytes.data(), enc_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("cross_attn_ref");
        auto *input = g.tensor({fx.hidden, fx.dec_seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        auto *encoder_input =
            g.tensor({fx.hidden, fx.enc_seq, fx.batch}, DataType::FP32)
                ->set_name("encoder_input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "cross_attn_mask",
            fx.enc_seq,
            fx.dec_seq,
            mask,
            mask_bytes);

        T5Attention attn(&g, "cross_attn", fx.config, true);
        attn.load(full_path);

        auto *output = attn.forward(input, encoder_input, mask);
        input->mark_input(true);
        encoder_input->mark_input(true);
        output->mark_output(true);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        runtime.bind_data(encoder_input, enc_data);
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

void cross_attention_backward_compare_ref(const CrossAttentionFixtureSpec &fx)
{
    const std::string full_path =
        cross_attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> enc_bytes =
        reader.read_tensor("encoder_input");
    std::vector<float> enc_data(enc_bytes.size() / sizeof(float));
    std::memcpy(enc_data.data(), enc_bytes.data(), enc_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<float> grad_input_result;
    {
        NNGraph g("cross_attn_bwd");
        auto *input =
            g.tensor({fx.hidden, fx.dec_seq, fx.batch}, DataType::FP32, true)
                ->set_name("input");
        auto *encoder_input =
            g.tensor({fx.hidden, fx.enc_seq, fx.batch}, DataType::FP32, true)
                ->set_name("encoder_input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "cross_attn_mask",
            fx.enc_seq,
            fx.dec_seq,
            mask,
            mask_bytes);

        T5Attention attn(&g, "cross_attn", fx.config, true);
        attn.load(full_path);

        auto *output = attn.forward(input, encoder_input, mask);

        input->mark_input(true);
        encoder_input->mark_input(true);
        output->mark_output(true);
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
        runtime.bind_data(encoder_input, enc_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result = runtime.get_output<float>(input->grad());
    }

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}

} // namespace

TEST_CASE("T5Attention cross forward builds output", "[model][t5]")
{
    CrossAttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            cross_attn_fixture_stem::t5_cross_attention, fx))
    {
        SKIP("Missing or invalid t5_cross_attention.json / .safetensors.");
    }
    NNGraph g("t5_cross_attn");
    T5Attention attn(&g, "cross_attn", fx.config, true);
    auto *input = g.tensor({fx.hidden, fx.dec_seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *encoder_input =
        g.tensor({fx.hidden, fx.enc_seq, fx.batch}, DataType::FP32)
            ->set_name("encoder_input");
    auto *output = attn.forward(input, encoder_input, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.hidden, fx.dec_seq, fx.batch}));
    REQUIRE(attn.parameters_recursive().size() == 4);
}

TEST_CASE("T5Attention cross load from safetensors roundtrip",
    "[model][t5][io]")
{
    CrossAttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            cross_attn_fixture_stem::t5_cross_attention, fx))
    {
        SKIP("Missing or invalid t5_cross_attention.json / .safetensors.");
    }
    const std::string data_path =
        cross_attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);

    NNGraph g1("load_graph");
    T5Attention attn1(&g1, "cross_attn", fx.config, true);
    attn1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_t5_cross_attn_roundtrip.safetensors";
    attn1.save(save_path);

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
    "T5Attention cross forward matches PyTorch reference",
    "[model][t5]")
{
    CrossAttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            cross_attn_fixture_stem::t5_cross_attention, fx))
    {
        SKIP("T5 cross-attention fixture not found.");
    }
    cross_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention cross backward matches PyTorch reference",
    "[model][t5]")
{
    CrossAttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            cross_attn_fixture_stem::t5_cross_attention, fx))
    {
        SKIP("T5 cross-attention fixture not found.");
    }
    cross_attention_backward_compare_ref(fx);
}

#endif
