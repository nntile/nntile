/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/t5/t5_attention.cc
 * Tests for T5Attention (self-attention).
 *
 * @version 1.1.0
 * */

#include "nntile/model/t5/t5_attention.hh"

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
    "T5Attention tests skipped (T5_DATA_DIR undefined)", "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace attn_fixture_stem
{

constexpr char t5_attention[] = "t5_attention";
constexpr char t5_attention_causal[] = "t5_attention_causal";

} // namespace attn_fixture_stem

namespace
{

using namespace nntile::test::t5_fixture;

struct AttentionFixtureSpec
{
    T5Config config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_attention_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    AttentionFixtureSpec &out)
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
        out.seq = json_index(j, "sequence_length");
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
    const char *stem, AttentionFixtureSpec &fx)
{
    const std::string dir = std::string(T5_DATA_DIR);
    if (!try_load_attention_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(t5_fixture_safetensors_path(dir, fx.stem));
    return st.good();
}

void t5_attention_forward_compare_ref(const AttentionFixtureSpec &fx)
{
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("attn_ref_") + fx.stem);
        auto *input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                          ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(
            g, reader, "attn_mask", fx.seq, fx.seq, mask, mask_bytes);

        T5Attention attn(&g, "attn", fx.config);
        attn.load(full_path);

        auto *output = attn.forward(input, nullptr, mask);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input, input_data);
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

void t5_attention_backward_compare_ref(const AttentionFixtureSpec &fx)
{
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<float> grad_input_result;
    {
        NNGraph g(std::string("attn_bwd_") + fx.stem);
        auto *input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32, true)
                          ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(
            g, reader, "attn_mask", fx.seq, fx.seq, mask, mask_bytes);

        T5Attention attn(&g, "attn", fx.config);
        attn.load(full_path);

        auto *output = attn.forward(input, nullptr, mask);

        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        output->backward();

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
        runtime.bind_data(input, input_data);
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

TEST_CASE("T5Attention forward builds output", "[model][t5]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("Missing or invalid t5_attention.json / .safetensors.");
    }
    NNGraph g("t5_attn");
    T5Attention attn(&g, "attn", fx.config);
    auto *input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                      ->set_name("input");
    auto *output = attn.forward(input, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.batch, fx.seq, fx.hidden}));
    REQUIRE(attn.parameters_recursive().size() == 4);
}

TEST_CASE("T5Attention load from safetensors roundtrip", "[model][t5][io]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("Missing or invalid t5_attention.json / .safetensors.");
    }
    const std::string data_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);

    NNGraph g1("load_graph");
    T5Attention attn1(&g1, "attn", fx.config);
    attn1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_t5_attn_roundtrip.safetensors";
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
    "T5Attention forward matches PyTorch reference (no mask)",
    "[model][t5][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("T5 attention fixture not found.");
    }
    t5_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention backward matches PyTorch reference (no mask)",
    "[model][t5][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("T5 attention fixture not found.");
    }
    t5_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention causal forward matches PyTorch reference",
    "[model][t5][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_causal, fx))
    {
        SKIP("T5 causal attention fixture not found.");
    }
    t5_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention causal backward matches PyTorch reference",
    "[model][t5][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_causal, fx))
    {
        SKIP("T5 causal attention fixture not found.");
    }
    t5_attention_backward_compare_ref(fx);
}

#endif
