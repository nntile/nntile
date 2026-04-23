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
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "context_fixture.hh"
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

static LlamaConfig test_config()
{
    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    return config;
}

static LlamaConfig test_config_gqa()
{
    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
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
    TensorGraph::Runtime& runtime, const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data("rope_sin", rope.sin_data);
    runtime.bind_data("rope_cos", rope.cos_data);
}

} // namespace

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    NNGraph g("llama_attn");
    auto config = test_config();

    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

TEST_CASE("LlamaAttention GQA forward builds output", "[model][llama][gqa]")
{
    NNGraph g("llama_attn_gqa");
    auto config = test_config_gqa();

    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("attn_ref");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(result.size() == ref_data.size());
    for(size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(std::abs(result[i] - ref_data[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention tiled matches untiled", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    const size_t input_nelems = input_bytes.size() / sizeof(float);
    std::vector<float> input_data(input_nelems);
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> untiled_result;
    {
        NNGraph g("llama_attn_untiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(output->name());
    }

    std::vector<float> tiled_result;
    {
        NNGraph g("llama_attn_tiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        for(auto* ag : tg.axis_groups())
        {
            // Do not tile head_size (breaks attention). Do not tile extent 2
            // (maxsumexp/logsumexp require dst.basetile_shape[0]==2).
            if(ag->extent != head_size && ag->extent != 2)
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(output->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention backward matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

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
        NNGraph g("attn_bwd");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    for(size_t i = 0; i < grad_input_result.size(); ++i)
    {
        REQUIRE(std::abs(grad_input_result[i] - grad_input_ref[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention backward tiled matches untiled", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes = reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(grad_out_data.data(), grad_out_bytes.data(),
        grad_out_bytes.size());

    std::vector<float> untiled_result;
    {
        NNGraph g("attn_bwd_untiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        untiled_result =
            runtime.get_output<float>(input->grad()->name());
    }

    std::vector<float> tiled_result;
    {
        NNGraph g("attn_bwd_tiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        for(auto* ag : tg.axis_groups())
        {
            if(ag->extent != head_size && ag->extent != 2)
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        tiled_result =
            runtime.get_output<float>(input->grad()->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA matches PyTorch reference", "[model][llama][gqa]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_gqa_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

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
        NNGraph g("attn_gqa_ref");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(result.size() == ref_data.size());
    for(size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(std::abs(result[i] - ref_data[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward matches PyTorch reference", "[model][llama][gqa]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_gqa_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

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
        NNGraph g("attn_gqa_bwd");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(g, reader, config, 4, 2, rope);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, nullptr);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    for(size_t i = 0; i < grad_input_result.size(); ++i)
    {
        REQUIRE(std::abs(grad_input_result[i] - grad_input_ref[i]) < tol);
    }
}
#endif
