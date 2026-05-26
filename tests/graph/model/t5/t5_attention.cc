/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/t5/t5_attention.cc
 * Tests for T5Attention (self-attention, ``sdpa_eager``).
 *
 * T5 has **no RoPE** in the graph; ``[norope]`` tags mark bundles that use
 * plain Q/K/V + SDPA only. Each reference bundle is a pair ``<stem>.json`` and
 * ``<stem>.safetensors`` from ``generate_test_data.py``.
 *
 * Catch tags:
 * ``[nomask]`` / ``[norope_nomask]`` — no ``attn_mask``;
 * ``[causal_mask]`` / ``[norope]`` — causal BOOL mask.
 *
 * Achieved relative Frobenius error (C++ vs PyTorch reference, seed 42):
 *
 * | Bundle | Forward | Backward |
 * |--------|---------|----------|
 * | no mask, no RoPE | ~1.2e-7 | ~2.3e-7 |
 * | causal mask, no RoPE | ~4.9e-8 | ~2.1e-7 |
 *
 * JSON tolerances: ``1e-6`` for both (see variant writer in
 * ``generate_test_data.py``).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_attention.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/t5/t5_config.hh"
#include "test_frobenius.hh"
#include "test_t5_attention_fixture.hh"
#include "test_t5_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::t5;
using namespace nntile::graph::io;

#ifndef T5_DATA_DIR

TEST_CASE(
    "T5Attention tests skipped (T5_DATA_DIR undefined)", "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::test::t5_attention_fixture;
using namespace nntile::test::t5_fixture;

void t5_attention_forward_compare_ref(const AttentionFixtureSpec& fx)
{
    const std::string full_path =
        attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("attn_ref_") + fx.stem);
        auto* input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(
            g, reader, "attn_mask", fx.seq, fx.seq, mask, mask_bytes);

        T5Attention attn(&g, "attn", fx.config);
        attn.load(full_path);

        auto* output = attn.forward(input, nullptr, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
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

void t5_attention_backward_compare_ref(const AttentionFixtureSpec& fx)
{
    const std::string full_path =
        attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);
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
        auto* input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32, true)
                          ->set_name("input");
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(
            g, reader, "attn_mask", fx.seq, fx.seq, mask, mask_bytes);

        T5Attention attn(&g, "attn", fx.config);
        attn.load(full_path);

        auto* output = attn.forward(input, nullptr, mask);

        input->mark_input(true);
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
    if(!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("Missing or invalid t5_attention.json / .safetensors.");
    }
    NNGraph g("t5_attn");
    T5Attention attn(&g, "attn", fx.config);
    auto* input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto* output = attn.forward(input, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
    REQUIRE(attn.parameters_recursive().size() == 4);
}

TEST_CASE("T5Attention load from safetensors roundtrip", "[model][t5][io]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::t5_attention, fx))
    {
        SKIP("Missing or invalid t5_attention.json / .safetensors.");
    }
    const std::string data_path =
        attention_fixture_safetensors_path(std::string(T5_DATA_DIR), fx);

    NNGraph g1("load_graph");
    T5Attention attn1(&g1, "attn", fx.config);
    attn1.load(data_path);

    const std::string save_path = "/tmp/nntile_t5_attn_roundtrip.safetensors";
    attn1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);
    for(const auto& name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        REQUIRE(reader.read_tensor(name) == reader2.read_tensor(name));
    }
    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention forward vs PyTorch (no mask, no RoPE)",
    "[model][t5][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_no_rope_nomask, fx))
    {
        SKIP("T5 attention no-RoPE / no-mask fixture pair not found.");
    }
    t5_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention backward vs PyTorch (no mask, no RoPE)",
    "[model][t5][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_no_rope_nomask, fx))
    {
        SKIP("T5 attention no-RoPE / no-mask fixture pair not found.");
    }
    t5_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention forward vs PyTorch (causal mask, no RoPE)",
    "[model][t5][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_no_rope_causal, fx))
    {
        SKIP("T5 attention no-RoPE / causal fixture pair not found.");
    }
    t5_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "T5Attention backward vs PyTorch (causal mask, no RoPE)",
    "[model][t5][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::t5_attention_no_rope_causal, fx))
    {
        SKIP("T5 attention no-RoPE / causal fixture pair not found.");
    }
    t5_attention_backward_compare_ref(fx);
}

#endif
