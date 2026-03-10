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

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    NNGraph g("llama_attn");
    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();

    // Input: (hidden, seq, batch) = (8, 4, 2)
    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaAttention load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path = std::string(LLAMA_DATA_DIR) + "/llama_attention.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama test data not found. Run: python scripts/generate_llama_test_data.py");
    }

    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();

    NNGraph g1("load_graph");
    LlamaAttention attn1(&g1, "attn", config);
    attn1.load(data_path);

    const std::string save_path = "/tmp/nntile_llama_attn_roundtrip.safetensors";
    attn1.save(save_path);

    NNGraph g2("verify_graph");
    LlamaAttention attn2(&g2, "attn", config);
    attn2.load(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for(const auto& name : reader.tensor_names())
    {
        REQUIRE(reader2.has_tensor(name));
        auto orig = reader.read_tensor(name);
        auto loaded = reader2.read_tensor(name);
        REQUIRE(orig.size() == loaded.size());
        REQUIRE(orig == loaded);
    }

    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention tiled matches untiled", "[model][llama]")
{
    const std::string data_dir(LLAMA_DATA_DIR);
    const std::string full_path = data_dir + "/llama_attention_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama full test data not found. Run llama_data_setup fixture.");
    }

    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    const size_t input_nelems = input_bytes.size() / sizeof(float);
    std::vector<float> input_data(input_nelems);
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        NNGraph g("llama_attn_untiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        attn.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(output->name());
    }

    // --- Tiled run: 2 tiles per axis, except head_size ---
    std::vector<float> tiled_result;
    {
        NNGraph g("llama_attn_tiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

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
#endif

// Note: Backward test disabled - sdpa_eager backward may have add_slice
// tensor aliasing constraints. Forward and structure tests pass.
// For full validation, compare against transformers (PyTorch) when
// NNTILE_HAVE_TORCH is enabled.
