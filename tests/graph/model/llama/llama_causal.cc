/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_causal.cc
 * Tests for LlamaCausal.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_causal.hh"
#include "nntile/graph/model/llama/llama_config.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;

static LlamaConfig test_config()
{
    LlamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    return config;
}

TEST_CASE("LlamaCausal forward builds output", "[model][llama]")
{
    NNGraph g("llama_causal");
    auto config = test_config();

    auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
    LlamaCausal model(&g, "model", config);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({100, 4, 2}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaCausal load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path =
        std::string(LLAMA_DATA_DIR) + "/llama_causal_full.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama causal test data not found.");
    }

    auto config = test_config();

    NNGraph g1("load_graph");
    LlamaCausal model1(&g1, "model", config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_causal_roundtrip.safetensors";
    model1.save(save_path);

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
    "LlamaCausal matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_causal_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama causal full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("causal_ref");
        auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
        LlamaCausal model(&g, "model", config);
        auto* output = model.forward(input_ids);
        input_ids->mark_input(true);
        output->mark_output(true);

        model.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
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
    "LlamaCausal tiled matches untiled", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_causal_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama causal full test data not found.");
    }

    auto config = test_config();

    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> untiled_result;
    {
        NNGraph g("causal_untiled");
        auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
        LlamaCausal model(&g, "model", config);
        auto* output = model.forward(input_ids);
        input_ids->mark_input(true);
        output->mark_output(true);

        model.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(output->name());
    }

    std::vector<float> tiled_result;
    {
        NNGraph g("causal_tiled");
        auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
        LlamaCausal model(&g, "model", config);
        auto* output = model.forward(input_ids);
        input_ids->mark_input(true);
        output->mark_output(true);

        model.load(full_path);

        auto axis_has_member = [](
            const graph::AxisDescriptor* ag,
            const std::string& name_substr, int axis_idx) -> bool
        {
            for(const auto& [node_ptr, axis] : ag->members)
            {
                if(axis != axis_idx)
                    continue;
                auto* node = static_cast<
                    const TensorGraph::TensorNode*>(node_ptr);
                if(node->name().find(name_substr) != std::string::npos)
                    return true;
            }
            return false;
        };

        TensorGraph& tg = g.tensor_graph();
        for(auto* ag : tg.axis_groups())
        {
            // head_size is axis 1 of q_weight (n_heads, head_size, n_emb)
            bool is_head_dim = axis_has_member(ag, "q_weight", 1);
            // vocab_size is axis 1 of embedding vocab (embed_dim, num_embeddings)
            bool is_vocab_dim = axis_has_member(ag, "vocab", 1);
            if(!is_head_dim && !is_vocab_dim && ag->extent > 2)
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
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
