/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_decoder.cc
 * Tests for LlamaDecoder.
 *
 * Each reference bundle is a **pair**: ``<stem>.json`` (``Llama`` fields for
 * ``LlamaConfig`` plus ``sequence_length``, ``batch``, tolerances) and
 * ``<stem>.safetensors`` (weights, ``rope_cos`` / ``rope_sin`` aligned with
 * PyTorch ``LlamaRotaryEmbedding``, input, reference tensors). Tests load the
 * JSON, build the decoder, ``load()`` weights, then bind RoPE tensors for
 * ``forward`` so the graph matches the HuggingFace reference path.
 * Pairs are produced by ``generate_test_data.py`` (``--block decoder`` /
 * ``decoder_gqa``).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_decoder.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "test_frobenius.hh"
#include "test_llama_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

#ifndef LLAMA_DATA_DIR

TEST_CASE(
    "LlamaDecoder tests skipped (LLAMA_DATA_DIR undefined)", "[model][llama]")
{
    SKIP("LLAMA_DATA_DIR not defined at compile time.");
}

#else

//! Basenames (no extension) for paired ``.json`` / ``.safetensors`` in
//! ``LLAMA_DATA_DIR`` — must match ``generate_test_data.py`` output.
namespace decoder_fixture_stem
{

constexpr char llama_decoder[] = "llama_decoder";
constexpr char llama_decoder_gqa[] = "llama_decoder_gqa";

} // namespace decoder_fixture_stem

namespace
{

using namespace nntile::test::llama_fixture;

//! Parsed ``<stem>.json`` (``version`` 2) next to ``<stem>.safetensors``.
struct DecoderFixtureSpec
{
    LlamaConfig config{};
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
        const auto &L = j.at("llama");
        out.config.hidden_size = json_index(L, "hidden_size");
        out.config.intermediate_size = json_index(L, "intermediate_size");
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

inline std::string decoder_fixture_safetensors_path(
    const std::string &data_dir, const DecoderFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

//! SKIP helper: JSON + safetensors must both exist and JSON must parse.
inline bool skip_unless_fixture_ready(const char *stem, DecoderFixtureSpec &fx)
{
    const std::string dir = std::string(LLAMA_DATA_DIR);
    if (!try_load_decoder_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(decoder_fixture_safetensors_path(dir, fx));
    return st.good();
}

void decoder_forward_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path =
        decoder_fixture_safetensors_path(data_dir, fx);
    const float tol = fx.forward_tol;
    const LlamaConfig &config = fx.config;
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;
    const Index hidden = fx.hidden;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("decoder_ref_") + fx.stem;
        NNGraph g(gname);
        auto *input = g.tensor({hidden, n_seq, n_batch}, DataType::FP32)
                          ->set_name("input");
        LlamaRopeInputs rope;
        REQUIRE(
            load_llama_rope_inputs(g, reader, config, n_seq, n_batch, rope));
        LlamaDecoder decoder(&g, "decoder", config);
        auto *output = decoder.forward(input, rope.sin, rope.cos, nullptr);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        decoder.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void decoder_backward_compare_ref(const DecoderFixtureSpec &fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path =
        decoder_fixture_safetensors_path(data_dir, fx);
    const float tol = fx.backward_tol;
    const LlamaConfig &config = fx.config;
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;
    const Index hidden = fx.hidden;

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
        const std::string gname = std::string("decoder_bwd_") + fx.stem;
        NNGraph g(gname);
        auto *input = g.tensor({hidden, n_seq, n_batch}, DataType::FP32, true)
                          ->set_name("input");
        LlamaRopeInputs rope;
        REQUIRE(
            load_llama_rope_inputs(g, reader, config, n_seq, n_batch, rope));
        LlamaDecoder decoder(&g, "decoder", config);
        auto *output = decoder.forward(input, rope.sin, rope.cos, nullptr);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        decoder.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        grad_input_result = runtime.get_output<float>(input->grad());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

} // namespace

TEST_CASE("LlamaDecoder forward builds output", "[model][llama]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::llama_decoder, fx))
    {
        SKIP("Missing or invalid llama_decoder.json / .safetensors.");
    }
    NNGraph g("llama_decoder");
    LlamaDecoder decoder(&g, "decoder", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = decoder.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("LlamaDecoder GQA forward builds output", "[model][llama][gqa]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::llama_decoder_gqa, fx))
    {
        SKIP("Missing or invalid llama_decoder_gqa.json / .safetensors.");
    }
    NNGraph g("llama_decoder_gqa");
    LlamaDecoder decoder(&g, "decoder", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = decoder.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("LlamaDecoder load from safetensors roundtrip", "[model][llama][io]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::llama_decoder, fx))
    {
        SKIP("Missing or invalid llama_decoder.json / .safetensors.");
    }
    const std::string data_path =
        decoder_fixture_safetensors_path(std::string(LLAMA_DATA_DIR), fx);

    NNGraph g1("load_graph");
    LlamaDecoder dec1(&g1, "decoder", fx.config);
    dec1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_decoder_roundtrip.safetensors";
    dec1.save(save_path);

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
    "LlamaDecoder matches PyTorch reference",
    "[model][llama]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::llama_decoder, fx))
    {
        SKIP("Llama decoder fixture pair not found.");
    }
    decoder_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaDecoder backward matches PyTorch reference",
    "[model][llama]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(decoder_fixture_stem::llama_decoder, fx))
    {
        SKIP("Llama decoder fixture pair not found.");
    }
    decoder_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaDecoder GQA matches PyTorch reference",
    "[model][llama][gqa]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::llama_decoder_gqa, fx))
    {
        SKIP("Llama decoder GQA fixture pair not found.");
    }
    decoder_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaDecoder GQA backward matches PyTorch reference",
    "[model][llama][gqa]")
{
    DecoderFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            decoder_fixture_stem::llama_decoder_gqa, fx))
    {
        SKIP("Llama decoder GQA fixture pair not found.");
    }
    decoder_backward_compare_ref(fx);
}

#endif
