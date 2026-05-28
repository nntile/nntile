/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/roberta/roberta_model.cc
 * Tests for RobertaModel.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/roberta/roberta_model.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/roberta/roberta_config.hh"
#include "test_frobenius.hh"
#include "test_roberta_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile::core;
using namespace nntile::graph;
using namespace nntile::graph::model::roberta;
using namespace nntile::graph::io;

#ifndef ROBERTA_DATA_DIR

TEST_CASE(
    "RobertaModel tests skipped (ROBERTA_DATA_DIR undefined)", "[model][roberta]")
{
    SKIP("ROBERTA_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::graph::test::roberta_fixture;

struct ModelFixtureSpec
{
    RobertaConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_model_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    ModelFixtureSpec &out)
{
    out = {};
    out.stem = stem_cstr;
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
        out.config.vocab_size = json_index(G, "vocab_size");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
        out.config.num_hidden_layers = json_index(G, "num_hidden_layers");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        out.config.layer_norm_eps = static_cast<float>(
            G.at("layer_norm_eps").get<double>());
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

inline std::string model_fixture_path(
    const std::string &data_dir, const ModelFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

} // namespace

TEST_CASE("RobertaModel forward builds output", "[model][roberta]")
{
    ModelFixtureSpec fx;
    if (!try_load_model_fixture_spec(
            std::string(ROBERTA_DATA_DIR), "roberta_model", fx))
    {
        SKIP("Missing roberta_model fixture.");
    }
    NNGraph g("roberta_model");
    RobertaModel model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
    auto *position_ids = g.tensor({fx.seq, fx.batch}, DataType::INT64)
                             ->set_name("position_ids");
    auto *output = model.forward(input_ids, position_ids, nullptr);
    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "RobertaModel forward matches PyTorch reference", "[model][roberta]")
{
    constexpr char stem[] = "roberta_model";
    ModelFixtureSpec fx;
    const std::string dir = std::string(ROBERTA_DATA_DIR);
    if (!try_load_model_fixture_spec(dir, stem, fx))
    {
        SKIP("RoBERTa model fixture not found.");
    }
    const std::string full_path = model_fixture_path(dir, fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("roberta_model_ref");
        auto *input_ids =
            g.tensor({fx.seq, fx.batch}, DataType::INT64)->set_name("input_ids");
        NNGraph::TensorNode *position_ids = nullptr;
        std::vector<std::int64_t> pos_data;
        REQUIRE(load_position_ids(
            g, reader, fx.seq, fx.batch, position_ids, pos_data));

        RobertaModel model(&g, "model", fx.config);
        model.load(full_path);
        auto *output = model.forward(input_ids, position_ids, nullptr);
        input_ids->mark_input(true);
        output->mark_output(true);
        mark_position_input(position_ids);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input_ids, ids_data);
        bind_position_input(runtime, position_ids, pos_data);
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
