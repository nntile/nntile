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
 * Tensor shapes (hidden, seq, batch) and head geometry match
 * ``wrappers/python/tests/model/test_llama_attention.py`` (``single_tile``):
 * head size 64, ``seq_len`` 64, ``n_batch`` 3; MHA (1,1) uses ``hidden=64``;
 * GQA (8,4) uses ``hidden=512``. Safetensors from
 * ``tests/graph/model/llama/generate_test_data.py`` (``ATTENTION_*_DIMS``).
 * RoPE/causal-mask reference bundles: default ``llama_attention(_gqa)_full`` plus
 * six extras from ``--write-attention-rope-mask-variants``. Catch tags:
 * ``[nomask]`` — no causal ``attn_mask`` (RoPE and no-RoPE bundles);
 * ``[causal_mask]`` — causal ``attn_mask``;
 * ``[norope]`` — no-RoPE bundles only (with or without causal mask);
 * ``[norope_nomask]`` — no-RoPE and no causal mask (subset of ``[nomask]``).
 * Run e.g. ``./test_llama_attention '[nomask]'`` or ``'[norope_nomask]'``.
 *
 * NNTile tensor **storage** is Fortran (column-major) everywhere, including
 * ``bind_hint`` bytes from safetensors (see ``generate_test_data.fortran_order``).
 * Attention linear weights in the fixtures are HuggingFace numerics reshaped to
 * the graph's 3D/4D layouts (no ``rotate_tensor_in`` step in the generator).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_attention.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/tensor/fill.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;
namespace gt = nntile::graph::tensor;

// Matches test_llama_attention.LlamaAttentionTestParams (single_tile).
static constexpr Index kAttnHeadSize = 64;
static constexpr Index kAttnSeq = 64;
static constexpr Index kAttnBatch = 3;

// MHA: 1 Q head, 1 KV head -> hidden = head_size.
static constexpr Index kMhaHidden = kAttnHeadSize;

// GQA: 8 Q heads, 4 KV heads -> hidden = 8 * head_size.
static constexpr Index kGqaNumHeads = 8;
static constexpr Index kGqaNumKv = 4;
static constexpr Index kGqaHidden = kAttnHeadSize * kGqaNumHeads;

static LlamaConfig test_config()
{
    LlamaConfig config;
    config.hidden_size = kMhaHidden;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    return config;
}

static LlamaConfig test_config_gqa()
{
    LlamaConfig config;
    config.hidden_size = kGqaHidden;
    config.num_attention_heads = kGqaNumHeads;
    config.num_key_value_heads = kGqaNumKv;
    config.compute_head_dim();
    return config;
}

namespace
{

//! Optional RoPE tensors from safetensors (same layout as Python LlamaAttention).
struct LlamaRopeInputs
{
    NNGraph::TensorNode* sin = nullptr;
    NNGraph::TensorNode* cos = nullptr;
    std::vector<float> sin_data;
    std::vector<float> cos_data;
};

inline bool load_llama_rope_inputs(
    NNGraph& g,
    const SafeTensorsReader& reader,
    const LlamaConfig& config,
    Index n_seq,
    Index n_batch,
    LlamaRopeInputs& out)
{
    out = {};
    if(!reader.has_tensor("rope_sin") || !reader.has_tensor("rope_cos"))
    {
        return false;
    }
    const Index head_dim = config.head_dim;
    if(head_dim % 2 != 0)
    {
        return false;
    }
    const Index half = head_dim / 2;
    out.sin = g.tensor({half, n_seq, n_batch}, "rope_sin", DataType::FP32);
    out.cos = g.tensor({half, n_seq, n_batch}, "rope_cos", DataType::FP32);
    auto read_f = [&](const char* name, std::vector<float>& dst)
    {
        std::vector<std::uint8_t> b = reader.read_tensor(name);
        dst.resize(b.size() / sizeof(float));
        std::memcpy(dst.data(), b.data(), b.size());
    };
    read_f("rope_sin", out.sin_data);
    read_f("rope_cos", out.cos_data);
    return true;
}

inline void mark_rope_inputs(const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    rope.sin->mark_input(true);
    rope.cos->mark_input(true);
}

inline void bind_rope_inputs(
    TileGraph::Runtime& runtime, const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data("rope_sin", rope.sin_data);
    runtime.bind_data("rope_cos", rope.cos_data);
}

//! Optional causal mask ``(seq, seq)`` for ``sdpa_eager`` (1 = keep logit).
//! Safetensors store float32 0/1 (``save_file`` maps numpy bool to F32); BOOL
//! is also accepted. Bytes bound to the graph are BOOL layout (1 byte/elem).
inline bool load_attn_mask_bool(
    NNGraph& g,
    const SafeTensorsReader& reader,
    Index n_seq,
    NNGraph::TensorNode*& out_mask,
    std::vector<std::uint8_t>& mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if(!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto& info = reader.tensor_info("attn_mask");
    if(info.shape.size() != 2 || info.shape[0] != n_seq
        || info.shape[1] != n_seq)
    {
        throw std::runtime_error("Llama attention test: attn_mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    out_mask = g.tensor({n_seq, n_seq}, "attn_mask", DataType::BOOL, false);
    auto raw = reader.read_tensor("attn_mask");
    if(info.dtype == DataType::BOOL)
    {
        if(raw.size() != n_el)
        {
            throw std::runtime_error(
                "Llama attention test: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if(info.dtype == DataType::FP32)
    {
        if(raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "Llama attention test: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto* p = reinterpret_cast<const float*>(raw.data());
        for(size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] =
                (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                              : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "Llama attention test: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(NNGraph::TensorNode* mask)
{
    if(mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(
    TileGraph::Runtime& runtime,
    NNGraph::TensorNode* mask,
    const std::vector<std::uint8_t>& mask_bytes)
{
    if(mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask->name(), mask_bytes);
}

} // namespace

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    NNGraph g("llama_attn");
    auto config = test_config();

    auto* input = g.tensor(
        {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({
        kMhaHidden, kAttnSeq, kAttnBatch}));
}

TEST_CASE("LlamaAttention GQA forward builds output", "[model][llama][gqa]")
{
    NNGraph g("llama_attn_gqa");
    auto config = test_config_gqa();

    auto* input = g.tensor(
        {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({
        kGqaHidden, kAttnSeq, kAttnBatch}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaAttention load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    auto config = test_config();

    NNGraph g1("load_graph");
    LlamaAttention attn1(&g1, "attn", config);
    attn1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_attn_roundtrip.safetensors";
    attn1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for(const auto& name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        auto orig = reader.read_tensor(name);
        auto loaded = reader2.read_tensor(name);
        REQUIRE(orig.size() == loaded.size());
        REQUIRE(orig == loaded);
    }

    std::remove(save_path.c_str());
}

namespace
{

//! MHA forward vs ``output_ref`` in ``full_path`` (must exist); ``fname`` names the graph.
void llama_mha_forward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("attn_ref_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void llama_mha_backward_vs_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes = reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(grad_out_data.data(), grad_out_bytes.data(),
        grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        const std::string gname = std::string("attn_bwd_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);

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

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

void llama_gqa_forward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config_gqa();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("attn_gqa_ref_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void llama_gqa_backward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config_gqa();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes = reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(grad_out_data.data(), grad_out_bytes.data(),
        grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        const std::string gname = std::string("attn_gqa_bwd_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);

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

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    const char* fname = "llama_attention_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    const char* fname = "llama_attention_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    const char* fname = "llama_attention_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    const char* fname = "llama_attention_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    const char* fname = "llama_attention_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    const char* fname = "llama_attention_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    const char* fname = "llama_attention_gqa_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_gqa_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    const char* fname = "llama_attention_gqa_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    const char* fname = "llama_attention_gqa_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    const char* fname = "llama_attention_gqa_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_gqa_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    const char* fname = "llama_attention_gqa_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    const char* fname = "llama_attention_gqa_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}
#endif
