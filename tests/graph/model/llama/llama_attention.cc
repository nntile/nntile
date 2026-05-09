/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_attention.cc
 * Tests for LlamaAttention (sdpa_eager-based).
 *
 * Each reference bundle is a **pair**: ``<stem>.json`` (``Llama`` attention
 * geometry, ``sequence_length``, ``batch``, tolerances) and
 * ``<stem>.safetensors`` (weights and reference tensors). Tests load the JSON,
 * build ``LlamaConfig``, construct ``LlamaAttention``, then ``load()`` the
 * sibling safetensors. Pairs are produced by ``generate_test_data.py``. Catch
 * tags:
 * ``[nomask]`` — no causal ``attn_mask`` (RoPE and no-RoPE bundles);
 * ``[causal_mask]`` — causal ``attn_mask``;
 * ``[norope]`` — no-RoPE bundles only (with or without causal mask);
 * ``[norope_nomask]`` — no-RoPE and no causal mask (subset of ``[nomask]``).
 *
 * NNTile tensor **storage** is Fortran (column-major) everywhere, including
 * ``bind_hint`` bytes from safetensors (see
 * ``generate_test_data.fortran_order``).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_attention.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "test_frobenius.hh"
#include "test_llama_attention_fixture.hh"
#include "test_llama_fixture_helpers.hh"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

#ifndef LLAMA_DATA_DIR

TEST_CASE("LlamaAttention tests skipped (LLAMA_DATA_DIR undefined)",
    "[model][llama]")
{
    SKIP("LLAMA_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::test::llama_fixture;
using namespace nntile::test::llama_attention_fixture;

void llama_attention_forward_compare_ref(const AttentionFixtureSpec &fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path =
        attention_fixture_safetensors_path(data_dir, fx);
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
        const std::string gname = std::string("attn_ref_") + fx.stem;
        NNGraph g(gname);
        auto *input = g.tensor({hidden, n_seq, n_batch}, DataType::FP32)
                          ->set_name("input");
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, n_seq, n_batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, n_seq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto *output = attn.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        attn.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void llama_attention_backward_compare_ref(const AttentionFixtureSpec &fx)
{
    const std::string data_dir = std::string(LLAMA_DATA_DIR);
    const std::string full_path =
        attention_fixture_safetensors_path(data_dir, fx);
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
        const std::string gname = std::string("attn_bwd_") + fx.stem;
        NNGraph g(gname);
        auto *input = g.tensor({hidden, n_seq, n_batch}, DataType::FP32, true)
                          ->set_name("input");
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, n_seq, n_batch, rope);
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, n_seq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto *output = attn.forward(input, rope.sin, rope.cos, mask);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
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
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

} // namespace

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention, fx))
    {
        SKIP("Missing or invalid llama_attention.json / .safetensors.");
    }
    NNGraph g("llama_attn");
    LlamaAttention attn(&g, "attn", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("LlamaAttention GQA forward builds output", "[model][llama][gqa]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention_gqa, fx))
    {
        SKIP("Missing or invalid llama_attention_gqa.json / .safetensors.");
    }
    NNGraph g("llama_attn_gqa");
    LlamaAttention attn(&g, "attn", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE(
    "LlamaAttention load from safetensors roundtrip", "[model][llama][io]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention, fx))
    {
        SKIP("Missing or invalid llama_attention.json / .safetensors.");
    }
    const std::string data_path =
        attention_fixture_safetensors_path(std::string(LLAMA_DATA_DIR), fx);

    NNGraph g1("load_graph");
    LlamaAttention attn1(&g1, "attn", fx.config);
    attn1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_attn_roundtrip.safetensors";
    attn1.save(save_path);

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
    "LlamaAttention MHA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_no_rope, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_causal, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_no_rope_causal, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_no_rope, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_causal, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_no_rope_causal, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention_gqa, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_no_rope, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_causal, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_no_rope_causal, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(attn_fixture_stem::llama_attention_gqa, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_no_rope, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_causal, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if (!skip_unless_fixture_ready(
            attn_fixture_stem::llama_attention_gqa_no_rope_causal, fx))
    {
        SKIP("Llama attention GQA fixture pair not found.");
    }
    llama_attention_backward_compare_ref(fx);
}

#endif
