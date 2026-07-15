/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/roberta/roberta_embeddings.cc
 * Tests for RobertaEmbeddings.
 *
 * @version 1.1.0
 * */

#include "nntile/model/roberta/roberta_embeddings.hh"

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
using namespace nntile::model::roberta;
using namespace nntile::io;

#ifndef ROBERTA_DATA_DIR

TEST_CASE(
    "RobertaEmbeddings tests skipped (ROBERTA_DATA_DIR undefined)",
    "[model][roberta]")
{
    SKIP("ROBERTA_DATA_DIR not defined at compile time.");
}

#else

namespace
{

using namespace nntile::test::roberta_fixture;

struct EmbFixtureSpec
{
    RobertaConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    std::string stem;
};

inline bool try_load_emb_spec(
    const std::string &data_dir, const char *stem, EmbFixtureSpec &out)
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
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.vocab_size = json_index(G, "vocab_size");
        out.config.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        out.config.layer_norm_eps = static_cast<float>(
            G.at("layer_norm_eps").get<double>());
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
    "RobertaEmbeddings forward matches PyTorch reference",
    "[model][roberta]")
{
    constexpr char stem[] = "roberta_embeddings";
    EmbFixtureSpec fx;
    const std::string dir = std::string(ROBERTA_DATA_DIR);
    if (!try_load_emb_spec(dir, stem, fx))
    {
        SKIP("RoBERTa embeddings fixture not found.");
    }
    const std::string full_path = dir + "/" + stem + ".safetensors";
    std::ifstream st(full_path);
    if (!st.good())
    {
        SKIP("RoBERTa embeddings safetensors not found.");
    }

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("emb_ref");
        auto *input_ids =
            g.tensor({fx.batch, fx.seq}, DataType::INT64)->set_name("input_ids");
        NNGraph::TensorNode *position_ids = nullptr;
        std::vector<std::int64_t> pos_data;
        REQUIRE(load_position_ids(
            g, reader, fx.seq, fx.batch, position_ids, pos_data));

        RobertaEmbeddings emb(&g, "embeddings", fx.config);
        emb.load(full_path);
        auto *output = emb.forward(input_ids, position_ids);
        mark_position_input(position_ids);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        g.bind_parameters(runtime);
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
