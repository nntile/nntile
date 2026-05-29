/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/roberta/roberta_layer.cc
 * Tests for BertLayer with RoBERTa reference weights.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_layer.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/roberta/roberta_config.hh"
#include "test_frobenius.hh"
#include "test_roberta_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile;
using namespace nntile::model::bert;
using namespace nntile::model::roberta;
using namespace nntile::io;

#ifndef ROBERTA_DATA_DIR

TEST_CASE(
    "RobertaLayer tests skipped (ROBERTA_DATA_DIR undefined)", "[model][roberta]")
{
    SKIP("ROBERTA_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::test::roberta_fixture;

struct LayerFixtureSpec
{
    BertConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    std::string stem;
};

inline bool try_load_layer_fixture_spec(
    const std::string &data_dir, const char *stem, LayerFixtureSpec &out)
{
    out = {};
    out.stem = stem;
    std::ifstream jf(data_dir + "/" + out.stem + ".json");
    if (!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        const auto &G = j.at("roberta");
        RobertaConfig rcfg;
        rcfg.hidden_size = json_index(G, "hidden_size");
        rcfg.intermediate_size = json_index(G, "intermediate_size");
        rcfg.num_attention_heads = json_index(G, "num_attention_heads");
        rcfg.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        rcfg.layer_norm_eps = static_cast<float>(
            G.at("layer_norm_eps").get<double>());
        out.config = to_bert_config(rcfg);
        out.hidden = out.config.hidden_size;
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.config.validate();
    }
    catch (...)
    {
        return false;
    }
    return true;
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RobertaLayer forward matches PyTorch reference", "[model][roberta]")
{
    constexpr char stem[] = "roberta_layer";
    LayerFixtureSpec fx;
    const std::string dir = std::string(ROBERTA_DATA_DIR);
    if (!try_load_layer_fixture_spec(dir, stem, fx))
    {
        SKIP("RoBERTa layer fixture not found.");
    }
    const std::string full_path = dir + "/" + stem + ".safetensors";
    std::ifstream st(full_path);
    if (!st.good())
    {
        SKIP("RoBERTa layer safetensors not found.");
    }

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("roberta_layer_ref");
        auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        BertLayer layer(&g, "layer", fx.config);
        layer.load(full_path);
        auto *output = layer.forward(input, nullptr);
        input->mark_input(true);
        output->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
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

#endif
