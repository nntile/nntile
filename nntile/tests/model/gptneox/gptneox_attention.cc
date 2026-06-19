/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/model/gptneox/gptneox_attention.cc
 * Tests for GptneoxAttention (sdpa_eager-based).
 *
 * Each reference bundle is a **pair**: ``<stem>.json`` (geometry, tolerances)
 * and ``<stem>.safetensors`` (weights and reference tensors). Pairs are
 * produced by ``generate_test_data.py``. Catch tags:
 * ``[nomask]`` — no causal ``attn_mask`` (RoPE and no-RoPE bundles);
 * ``[causal_mask]`` — causal ``attn_mask``;
 * ``[norope]`` — no-RoPE bundles only (with or without causal mask);
 * ``[norope_nomask]`` — no-RoPE and no causal mask (subset of ``[nomask]``).
 *
 * Run subsets, e.g. ``ctest -R gptneox_attention --extra-tests-args
 * '[nomask]'`` or ``'[norope]'`` / ``'~[causal_mask]'``, to see whether
 * disabling mask or RoPE reaches tight relative error.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_attention.hh"

#include "context_fixture.hh"
#include "test_runtime_bind_helpers.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/gptneox/gptneox_config.hh"
#include "test_frobenius.hh"
#include "test_gptneox_attention_fixture.hh"
#include "test_gptneox_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile;
using namespace nntile::model::gptneox;
using namespace nntile::io;

#ifndef GPTNEOX_DATA_DIR

TEST_CASE(
    "GptneoxAttention tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::test::gptneox_attention_fixture;
using namespace nntile::test::gptneox_fixture;

struct AttentionRunContext
{
    GptneoxRopeInputs rope{};
    NNGraph::TensorNode* mask = nullptr;
    std::vector<std::uint8_t> mask_bytes;
};

inline void prepare_attention_run(NNGraph& g,
    const SafeTensorsReader& reader,
    const AttentionFixtureSpec& fx,
    AttentionRunContext& ctx)
{
    ctx = {};
    load_gptneox_rope_inputs(
        g, reader, fx.config, fx.seq, fx.batch, ctx.rope);
    load_attn_mask_bool(g, reader, fx.seq, ctx.mask, ctx.mask_bytes);
}

inline NNGraph::TensorNode* run_attention_forward(NNGraph& g,
    NNGraph::TensorNode* input,
    const AttentionFixtureSpec& fx,
    const AttentionRunContext& ctx,
    const std::string& weights_path)
{
    GptneoxAttention attn(&g, "attn", fx.config);
    attn.load(weights_path);
    return attn.forward(input, ctx.rope.sin, ctx.rope.cos, ctx.mask);
}

inline void bind_attention_runtime_inputs(Runtime& runtime,
    NNGraph::TensorNode* input,
    const std::vector<float>& input_data,
    const AttentionRunContext& ctx)
{
    runtime.bind_data(input, input_data);
    bind_rope_inputs(runtime, ctx.rope);
    bind_mask_input(runtime, ctx.mask, ctx.mask_bytes);
}

void gptneox_attention_forward_compare_ref(const AttentionFixtureSpec& fx)
{
    const std::string full_path =
        attention_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("attn_ref_") + fx.stem);
        auto* input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                          ->set_name("input");
        AttentionRunContext ctx;
        prepare_attention_run(g, reader, fx, ctx);
        auto* output = run_attention_forward(g, input, fx, ctx, full_path);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(ctx.rope);
        mark_mask_input(ctx.mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        nntile::test::bind_hints_from_tensor_graph(runtime, g.tensor_graph());
        bind_attention_runtime_inputs(runtime, input, input_data, ctx);
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

void gptneox_attention_backward_compare_ref(const AttentionFixtureSpec& fx)
{
    const std::string full_path =
        attention_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);
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
        auto* input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32, true)
                          ->set_name("input");
        AttentionRunContext ctx;
        prepare_attention_run(g, reader, fx, ctx);
        auto* output = run_attention_forward(g, input, fx, ctx, full_path);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(ctx.rope);
        mark_mask_input(ctx.mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        nntile::test::bind_hints_from_tensor_graph(runtime, g.tensor_graph());
        bind_attention_runtime_inputs(runtime, input, input_data, ctx);
        runtime.bind_data(grad_output_tensor, grad_out_data);
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

TEST_CASE("GptneoxAttention forward builds output", "[model][gptneox]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::gptneox_attention, fx))
    {
        SKIP("Missing or invalid gptneox_attention.json / .safetensors.");
    }
    NNGraph g("gptneox_attn");
    GptneoxAttention attn(&g, "attn", fx.config);
    auto* input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                      ->set_name("input");
    auto* output = attn.forward(input, nullptr, nullptr, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.batch, fx.seq, fx.hidden}));
    REQUIRE(attn.parameters_recursive().size() == 4);
}

TEST_CASE("GptneoxAttention load from safetensors roundtrip", "[model][gptneox][io]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::gptneox_attention, fx))
    {
        SKIP("Missing or invalid gptneox_attention.json / .safetensors.");
    }
    const std::string data_path =
        attention_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoxAttention attn1(&g1, "attn", fx.config);
    attn1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneox_attn_roundtrip.safetensors";
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
    "GptneoxAttention forward vs PyTorch (no causal mask, RoPE)",
    "[model][gptneox][nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::gptneox_attention, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention forward vs PyTorch (no causal mask, no RoPE)",
    "[model][gptneox][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_no_rope, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention forward vs PyTorch (causal mask, RoPE)",
    "[model][gptneox][causal_mask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_causal, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention forward vs PyTorch (causal mask, no RoPE)",
    "[model][gptneox][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_no_rope_causal, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention backward vs PyTorch (no causal mask, RoPE)",
    "[model][gptneox][nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::gptneox_attention, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention backward vs PyTorch (no causal mask, no RoPE)",
    "[model][gptneox][nomask][norope][norope_nomask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_no_rope, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention backward vs PyTorch (causal mask, RoPE)",
    "[model][gptneox][causal_mask]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_causal, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_backward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoxAttention backward vs PyTorch (causal mask, no RoPE)",
    "[model][gptneox][causal_mask][norope]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(
            attn_fixture_stem::gptneox_attention_no_rope_causal, fx))
    {
        SKIP("GPT-NeoX attention fixture pair not found.");
    }
    gptneox_attention_backward_compare_ref(fx);
}

#endif
