/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/gpt2/gpt2_block.cc
 * Tests for Gpt2Block.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_block.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/gpt2/gpt2_config.hh"
#include "test_frobenius.hh"
#include "test_gpt2_fixture_helpers.hh"

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
using namespace nntile::model::gpt2;
using namespace nntile::io;

#ifndef GPT2_DATA_DIR

TEST_CASE(
    "Gpt2Block tests skipped (GPT2_DATA_DIR undefined)", "[model][gpt2]")
{
    SKIP("GPT2_DATA_DIR not defined at compile time.");
}

#else

namespace block_fixture_stem
{

constexpr char gpt2_block[] = "gpt2_block";

} // namespace block_fixture_stem

namespace
{

using namespace nntile::test::gpt2_fixture;

struct BlockFixtureSpec
{
    Gpt2Config config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_block_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    BlockFixtureSpec &out)
{
    out = {};
    out.stem = stem_cstr;
    const std::string jpath = data_dir + "/" + out.stem + ".json";
    std::ifstream jf(jpath);
    if (!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        if (j.at("version").get<int>() != 2)
        {
            return false;
        }
        if (j.at("stem").get<std::string>() != out.stem)
        {
            return false;
        }
        const auto &G = j.at("gpt2");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        out.hidden = out.config.hidden_size;
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
        out.config.validate();
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline std::string block_fixture_safetensors_path(
    const std::string &data_dir, const BlockFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, BlockFixtureSpec &fx)
{
    const std::string dir = std::string(GPT2_DATA_DIR);
    if (!try_load_block_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(block_fixture_safetensors_path(dir, fx));
    return st.good();
}

void block_forward_compare_ref(const BlockFixtureSpec &fx)
{
    const std::string full_path =
        block_fixture_safetensors_path(std::string(GPT2_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("block_ref_") + fx.stem);
        auto *input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                          ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);

        Gpt2Block block(&g, "block", fx.config);
        block.load(full_path);

        auto *output = block.forward(input, mask);
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

void block_backward_compare_ref(const BlockFixtureSpec &fx)
{
    const std::string full_path =
        block_fixture_safetensors_path(std::string(GPT2_DATA_DIR), fx);
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
        NNGraph g(std::string("block_bwd_") + fx.stem);
        auto *input =
            g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32, true)
                ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);

        Gpt2Block block(&g, "block", fx.config);
        block.load(full_path);
        auto *output = block.forward(input, mask);

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

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}

} // namespace

TEST_CASE("Gpt2Block forward builds output", "[model][gpt2]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::gpt2_block, fx))
    {
        SKIP("Missing or invalid gpt2_block.json / .safetensors.");
    }
    NNGraph g("gpt2_block");
    Gpt2Block block(&g, "block", fx.config);
    auto *input = g.tensor({fx.batch, fx.seq, fx.hidden}, DataType::FP32)
                      ->set_name("input");
    auto *position_ids = g.tensor({fx.batch, fx.seq}, DataType::INT64)
                             ->set_name("position_ids");
    (void)position_ids;
    NNGraph::TensorNode *mask = nullptr;
    auto *output = block.forward(input, mask);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.batch, fx.seq, fx.hidden}));
}

TEST_CASE("Gpt2Block load from safetensors roundtrip", "[model][gpt2][io]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::gpt2_block, fx))
    {
        SKIP("Missing or invalid gpt2_block.json / .safetensors.");
    }
    const std::string data_path =
        block_fixture_safetensors_path(std::string(GPT2_DATA_DIR), fx);

    NNGraph g1("load_graph");
    Gpt2Block block1(&g1, "block", fx.config);
    block1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gpt2_block_roundtrip.safetensors";
    block1.save(save_path);

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
    "Gpt2Block forward matches PyTorch reference", "[model][gpt2]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::gpt2_block, fx))
    {
        SKIP("GPT-2 block fixture not found.");
    }
    block_forward_compare_ref(fx);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Gpt2Block backward matches PyTorch reference",
    "[model][gpt2]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::gpt2_block, fx))
    {
        SKIP("GPT-2 block fixture not found.");
    }
    block_backward_compare_ref(fx);
}

#endif
