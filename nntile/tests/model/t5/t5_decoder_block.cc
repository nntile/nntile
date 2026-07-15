/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/t5/t5_decoder_block.cc
 * Tests for T5DecoderBlock.
 *
 * @version 1.1.0
 * */

#include "nntile/model/t5/t5_block.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/t5/t5_config.hh"
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
using namespace nntile;
using namespace nntile::model::t5;
using namespace nntile::io;

#ifndef T5_DATA_DIR

TEST_CASE(
    "T5DecoderBlock tests skipped (T5_DATA_DIR undefined)", "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace decoder_fixture_stem
{

constexpr char t5_decoder_block[] = "t5_decoder_block";

} // namespace decoder_fixture_stem

namespace
{

using namespace nntile::test::t5_fixture;

struct DecoderBlockFixtureSpec
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

inline bool try_load_decoder_block_fixture_spec(
    const std::string &data_dir,
    const char *stem_cstr,
    DecoderBlockFixtureSpec &out)
{
    out = {};
    nlohmann::json j;
    if (!try_open_t5_fixture_json(data_dir, stem_cstr, out.stem, j))
    {
        return false;
    }
    try
    {
        load_t5_config_from_fixture_json(j, out.config);
        out.hidden = out.config.d_model;
        out.enc_seq = json_index(j, "encoder_sequence_length");
        out.dec_seq = json_index(j, "decoder_sequence_length");
        out.batch = json_index(j, "batch");
        load_t5_fixture_tolerances(j, out.forward_tol, out.backward_tol);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline bool skip_unless_fixture_ready(
    const char *stem, DecoderBlockFixtureSpec &fx)
{
    const std::string dir = std::string(T5_DATA_DIR);
    if (!try_load_decoder_block_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(t5_fixture_safetensors_path(dir, fx.stem));
    return st.good();
}

void decoder_block_forward_compare_ref(const DecoderBlockFixtureSpec &fx)
{
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> enc_bytes =
        reader.read_tensor("encoder_hidden_states");
    std::vector<float> enc_data(enc_bytes.size() / sizeof(float));
    std::memcpy(enc_data.data(), enc_bytes.data(), enc_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("decoder_ref");
        auto *input = g.tensor({fx.batch, fx.dec_seq, fx.hidden}, DataType::FP32)
                          ->set_name("input");
        auto *encoder_hidden =
            g.tensor({fx.batch, fx.enc_seq, fx.hidden}, DataType::FP32)
                ->set_name("encoder_hidden_states");
        NNGraph::TensorNode *decoder_mask = nullptr;
        std::vector<std::uint8_t> decoder_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "decoder_attn_mask",
            fx.dec_seq,
            fx.dec_seq,
            decoder_mask,
            decoder_mask_bytes);
        NNGraph::TensorNode *cross_mask = nullptr;
        std::vector<std::uint8_t> cross_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "cross_attn_mask",
            fx.enc_seq,
            fx.dec_seq,
            cross_mask,
            cross_mask_bytes);

        T5DecoderBlock decoder(&g, "decoder", fx.config);
        decoder.load(full_path);

        auto *output = decoder.forward(input,
            encoder_hidden,
            decoder_mask,
            cross_mask);
        mark_mask_input(decoder_mask);
        mark_mask_input(cross_mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input, input_data);
        runtime.bind_data(encoder_hidden, enc_data);
        bind_mask_input(runtime, decoder_mask, decoder_mask_bytes);
        bind_mask_input(runtime, cross_mask, cross_mask_bytes);
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

void decoder_block_backward_compare_ref(const DecoderBlockFixtureSpec &fx)
{
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> enc_bytes =
        reader.read_tensor("encoder_hidden_states");
    std::vector<float> enc_data(enc_bytes.size() / sizeof(float));
    std::memcpy(enc_data.data(), enc_bytes.data(), enc_bytes.size());

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
        NNGraph g("decoder_bwd");
        auto *input =
            g.tensor({fx.batch, fx.dec_seq, fx.hidden}, DataType::FP32, true)
                ->set_name("input");
        auto *encoder_hidden =
            g.tensor({fx.batch, fx.enc_seq, fx.hidden}, DataType::FP32, true)
                ->set_name("encoder_hidden_states");
        NNGraph::TensorNode *decoder_mask = nullptr;
        std::vector<std::uint8_t> decoder_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "decoder_attn_mask",
            fx.dec_seq,
            fx.dec_seq,
            decoder_mask,
            decoder_mask_bytes);
        NNGraph::TensorNode *cross_mask = nullptr;
        std::vector<std::uint8_t> cross_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "cross_attn_mask",
            fx.enc_seq,
            fx.dec_seq,
            cross_mask,
            cross_mask_bytes);

        T5DecoderBlock decoder(&g, "decoder", fx.config);
        decoder.load(full_path);
        auto *output = decoder.forward(input,
            encoder_hidden,
            decoder_mask,
            cross_mask);

        mark_mask_input(decoder_mask);
        mark_mask_input(cross_mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        output->backward();

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input, input_data);
        runtime.bind_data(encoder_hidden, enc_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_mask_input(runtime, decoder_mask, decoder_mask_bytes);
        bind_mask_input(runtime, cross_mask, cross_mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result = runtime.get_output<float>(input->grad());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}

} // namespace

TEST_CASE("T5DecoderBlock forward builds output", "[model][t5]")
{
    DecoderBlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::t5_decoder_block, fx))
    {
        SKIP("Missing or invalid t5_decoder_block.json / .safetensors.");
    }
    NNGraph g("t5_decoder_block");
    T5DecoderBlock decoder(&g, "decoder", fx.config);
    auto *input = g.tensor({fx.batch, fx.dec_seq, fx.hidden}, DataType::FP32)
                      ->set_name("input");
    auto *encoder_hidden =
        g.tensor({fx.batch, fx.enc_seq, fx.hidden}, DataType::FP32)
            ->set_name("encoder_hidden_states");
    auto *output = decoder.forward(input, encoder_hidden, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.batch, fx.dec_seq, fx.hidden}));
}

TEST_CASE("T5DecoderBlock load from safetensors roundtrip", "[model][t5][io]")
{
    DecoderBlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::t5_decoder_block, fx))
    {
        SKIP("Missing or invalid t5_decoder_block.json / .safetensors.");
    }
    const std::string data_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);

    NNGraph g1("load_graph");
    T5DecoderBlock decoder1(&g1, "decoder", fx.config);
    decoder1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_t5_decoder_block_roundtrip.safetensors";
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5DecoderBlock forward matches PyTorch reference", "[model][t5]")
{
    DecoderBlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::t5_decoder_block, fx))
    {
        SKIP("T5 decoder block fixture not found.");
    }
    decoder_block_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5DecoderBlock backward matches PyTorch reference",
    "[model][t5]")
{
    DecoderBlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::t5_decoder_block, fx))
    {
        SKIP("T5 decoder block fixture not found.");
    }
    decoder_block_backward_compare_ref(fx);
}

#endif
