/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/bert/bert_layer.cc
 * Tests for BertLayer.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_layer.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/bert/bert_config.hh"
#include "test_frobenius.hh"
#include "test_bert_fixture_helpers.hh"

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
using namespace nntile::model::bert;
using namespace nntile::io;

#ifndef BERT_DATA_DIR

TEST_CASE(
    "BertLayer tests skipped (BERT_DATA_DIR undefined)", "[model][bert]")
{
    SKIP("BERT_DATA_DIR not defined at compile time.");
}

#else

namespace block_fixture_stem
{

constexpr char bert_layer[] = "bert_layer";

} // namespace block_fixture_stem

namespace
{

using namespace nntile::test::bert_fixture;

struct BlockFixtureSpec
{
    BertConfig config{};
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
        const auto &G = j.at("bert");
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
    const std::string dir = std::string(BERT_DATA_DIR);
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
        block_fixture_safetensors_path(std::string(BERT_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g(std::string("block_ref_") + fx.stem);
        auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);

        BertLayer block(&g, "layer", fx.config);
        block.load(full_path);

        auto *output = block.forward(input, mask);
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

void block_backward_compare_ref(const BlockFixtureSpec &fx)
{
    const std::string full_path =
        block_fixture_safetensors_path(std::string(BERT_DATA_DIR), fx);
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
            g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32, true)
                ->set_name("input");
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes);

        BertLayer block(&g, "layer", fx.config);
        block.load(full_path);
        auto *output = block.forward(input, mask);

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

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}


} // namespace

TEST_CASE("BertLayer forward builds output", "[model][bert]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::bert_layer, fx))
    {
        SKIP("Missing or invalid bert_layer.json / .safetensors.");
    }
    NNGraph g("bert_layer");
    BertLayer block(&g, "layer", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    NNGraph::TensorNode *mask = nullptr;
    auto *output = block.forward(input, mask);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE("BertLayer load from safetensors roundtrip", "[model][bert][io]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::bert_layer, fx))
    {
        SKIP("Missing or invalid bert_layer.json / .safetensors.");
    }
    const std::string data_path =
        block_fixture_safetensors_path(std::string(BERT_DATA_DIR), fx);

    NNGraph g1("load_graph");
    BertLayer block1(&g1, "layer", fx.config);
    block1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_bert_layer_roundtrip.safetensors";
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
    "BertLayer forward matches PyTorch reference", "[model][bert]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::bert_layer, fx))
    {
        SKIP("BERT block fixture not found.");
    }
    block_forward_compare_ref(fx);
}


TEST_CASE_METHOD(nntile::test::ContextFixture,
    "BertLayer backward matches PyTorch reference",
    "[model][bert]")
{
    BlockFixtureSpec fx;
    if (!skip_unless_fixture_ready(block_fixture_stem::bert_layer, fx))
    {
        SKIP("BERT block fixture not found.");
    }
    block_backward_compare_ref(fx);
}

#endif
