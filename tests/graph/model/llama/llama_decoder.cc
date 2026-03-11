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
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/model/llama/llama_decoder.hh"
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
    config.intermediate_size = 16;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    return config;
}

static LlamaConfig test_config_gqa()
{
    LlamaConfig config;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.compute_head_dim();
    return config;
}

TEST_CASE("LlamaDecoder forward builds output", "[model][llama]")
{
    NNGraph g("llama_decoder");
    auto config = test_config();

    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaDecoder decoder(&g, "decoder", config);
    auto* output = decoder.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

TEST_CASE("LlamaDecoder GQA forward builds output", "[model][llama][gqa]")
{
    NNGraph g("llama_decoder_gqa");
    auto config = test_config_gqa();

    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaDecoder decoder(&g, "decoder", config);
    auto* output = decoder.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaDecoder load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_full.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama decoder test data not found.");
    }

    auto config = test_config();

    NNGraph g1("load_graph");
    LlamaDecoder dec1(&g1, "decoder", config);
    dec1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_llama_decoder_roundtrip.safetensors";
    dec1.save(save_path);

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
    "LlamaDecoder matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder full test data not found.");
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
        NNGraph g("decoder_ref");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
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
    "LlamaDecoder tiled matches untiled", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder full test data not found.");
    }

    auto config = test_config();
    const Index head_size = config.head_dim;

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> untiled_result;
    {
        NNGraph g("decoder_untiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(output->name());
    }

    std::vector<float> tiled_result;
    {
        NNGraph g("decoder_tiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        decoder.load(full_path);

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
    "LlamaDecoder backward matches PyTorch reference", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder full test data not found.");
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
        NNGraph g("decoder_bwd");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
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
    "LlamaDecoder backward tiled matches untiled", "[model][llama]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder full test data not found.");
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
        NNGraph g("decoder_bwd_untiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        runtime.execute();
        runtime.wait();

        untiled_result =
            runtime.get_output<float>(input->grad()->name());
    }

    std::vector<float> tiled_result;
    {
        NNGraph g("decoder_bwd_tiled");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        decoder.load(full_path);

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
    "LlamaDecoder GQA matches PyTorch reference", "[model][llama][gqa]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_gqa_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder GQA test data not found.");
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
        NNGraph g("decoder_gqa_ref");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
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
    "LlamaDecoder GQA backward matches PyTorch reference", "[model][llama][gqa]")
{
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/llama_decoder_gqa_full.safetensors";
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama decoder GQA backward test data not found.");
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
        NNGraph g("decoder_gqa_bwd");
        auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32, true);
        LlamaDecoder decoder(&g, "decoder", config);
        auto* output = decoder.forward(input, nullptr, nullptr, nullptr);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        decoder.load(full_path);

        TensorGraph& tg = g.tensor_graph();
        TensorGraph::Runtime runtime(tg);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
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
