/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_model.cc
 * Tests for LlamaModel.
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
#include "test_frobenius.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/model/llama/llama_model.hh"
#include "nntile/graph/tensor/fill.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;
namespace gt = nntile::graph::tensor;

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

static LlamaConfig test_config_gqa()
{
    LlamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.compute_head_dim();
    return config;
}

TEST_CASE("LlamaModel forward builds output", "[model][llama]")
{
    NNGraph g("llama_model");
    auto config = test_config();

    auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
    LlamaModel model(&g, "model", config);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

TEST_CASE("LlamaModel GQA forward builds output", "[model][llama][gqa]")
{
    NNGraph g("llama_model_gqa");
    auto config = test_config_gqa();

    auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
    LlamaModel model(&g, "model", config);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaModel load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path =
        std::string(LLAMA_DATA_DIR) + "/llama_model_full.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama model test data not found.");
    }

    auto config = test_config();

    NNGraph g1("load_graph");
    LlamaModel model1(&g1, "model", config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_model_roundtrip.safetensors";
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
    "LlamaModel matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_model_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama model full test data not found.");
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
        NNGraph g("model_ref");
        auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
        LlamaModel model(&g, "model", config);
        auto* output = model.forward(input_ids);
        input_ids->mark_input(true);
        output->mark_output(true);

        model.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaModel GQA matches PyTorch reference", "[model][llama][gqa]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_model_gqa_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama model GQA test data not found.");
    }

    auto config = test_config_gqa();

    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("model_gqa_ref");
        auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
        LlamaModel model(&g, "model", config);
        auto* output = model.forward(input_ids);
        input_ids->mark_input(true);
        output->mark_output(true);

        model.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input_ids", ids_data);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    constexpr float tol = 1e-4f;
    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}
#endif
