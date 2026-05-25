/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/t5/t5_conditional.cc
 * Tests for T5ForConditionalGeneration.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_for_conditional_generation.hh"

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
    "T5ForConditionalGeneration tests skipped (T5_DATA_DIR undefined)",
    "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace conditional_fixture_stem
{

constexpr char t5_conditional[] = "t5_conditional";

} // namespace conditional_fixture_stem

namespace
{

using namespace nntile::test::t5_fixture;

struct ConditionalFixtureSpec
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

inline bool try_load_conditional_fixture_spec(
    const std::string &data_dir,
    const char *stem_cstr,
    ConditionalFixtureSpec &out)
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

inline std::string conditional_fixture_safetensors_path(
    const std::string &data_dir, const ConditionalFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(
    const char *stem, ConditionalFixtureSpec &fx)
{
    const std::string dir = std::string(T5_DATA_DIR);
    if (!try_load_conditional_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(conditional_fixture_safetensors_path(dir, fx));
    return st.good();
}

} // namespace

TEST_CASE("T5ForConditionalGeneration forward builds output", "[model][t5]")
{
    ConditionalFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            conditional_fixture_stem::t5_conditional, fx))
    {
        SKIP("Missing or invalid t5_conditional.json / .safetensors.");
    }
    NNGraph g("t5_conditional");
    T5ForConditionalGeneration conditional(&g, "conditional", fx.config);
    auto *encoder_input_ids =
        g.tensor({fx.enc_seq, fx.batch}, DataType::INT64)
            ->set_name("encoder_input_ids");
    auto *decoder_input_ids =
        g.tensor({fx.dec_seq, fx.batch}, DataType::INT64)
            ->set_name("decoder_input_ids");
    auto *output = conditional.forward(
        encoder_input_ids, decoder_input_ids, nullptr, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({fx.config.vocab_size,
        fx.dec_seq,
        fx.batch}));
}

TEST_CASE("T5ForConditionalGeneration load from safetensors roundtrip",
    "[model][t5][io]")
{
    ConditionalFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            conditional_fixture_stem::t5_conditional, fx))
    {
        SKIP("Missing or invalid t5_conditional.json / .safetensors.");
    }
    const std::string data_path =
        conditional_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);

    NNGraph g1("load_graph");
    T5ForConditionalGeneration conditional1(&g1, "conditional", fx.config);
    conditional1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_t5_conditional_roundtrip.safetensors";
    conditional1.save(save_path);

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
    "T5ForConditionalGeneration forward matches PyTorch reference",
    "[model][t5]")
{
    ConditionalFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            conditional_fixture_stem::t5_conditional, fx))
    {
        SKIP("T5 conditional fixture not found.");
    }
    const std::string full_path =
        conditional_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> enc_ids_bytes =
        reader.read_tensor("encoder_input_ids");
    std::vector<std::int64_t> enc_ids_data(
        enc_ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(
        enc_ids_data.data(), enc_ids_bytes.data(), enc_ids_bytes.size());

    std::vector<std::uint8_t> dec_ids_bytes =
        reader.read_tensor("decoder_input_ids");
    std::vector<std::int64_t> dec_ids_data(
        dec_ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(
        dec_ids_data.data(), dec_ids_bytes.data(), dec_ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("conditional_ref");
        auto *encoder_input_ids =
            g.tensor({fx.enc_seq, fx.batch}, DataType::INT64)
                ->set_name("encoder_input_ids");
        auto *decoder_input_ids =
            g.tensor({fx.dec_seq, fx.batch}, DataType::INT64)
                ->set_name("decoder_input_ids");
        NNGraph::TensorNode *decoder_mask = nullptr;
        std::vector<std::uint8_t> decoder_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "decoder_attention_mask",
            fx.dec_seq,
            fx.dec_seq,
            decoder_mask,
            decoder_mask_bytes);
        NNGraph::TensorNode *cross_mask = nullptr;
        std::vector<std::uint8_t> cross_mask_bytes;
        load_attn_mask_bool(g,
            reader,
            "cross_attention_mask",
            fx.enc_seq,
            fx.dec_seq,
            cross_mask,
            cross_mask_bytes);

        T5ForConditionalGeneration conditional(
            &g, "conditional", fx.config);
        conditional.load(full_path);

        auto *output = conditional.forward(encoder_input_ids,
            decoder_input_ids,
            nullptr,
            decoder_mask,
            cross_mask);
        encoder_input_ids->mark_input(true);
        decoder_input_ids->mark_input(true);
        output->mark_output(true);
        mark_mask_input(decoder_mask);
        mark_mask_input(cross_mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(encoder_input_ids, enc_ids_data);
        runtime.bind_data(decoder_input_ids, dec_ids_data);
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

#endif
