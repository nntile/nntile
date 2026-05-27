/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneox/gptneox_config.cc
 * Tests for GptneoxConfig.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "nntile/graph/model/gptneox/gptneox_config.hh"
#include "nntile/graph/model/gptneox/gptneox_config_json.hh"

using nntile::core::Index;
using namespace nntile::graph::model::gptneox;

TEST_CASE("GptneoxConfig default values", "[model][gptneox]")
{
    GptneoxConfig config;
    REQUIRE(config.vocab_size == 50280);
    REQUIRE(config.hidden_size == 1024);
    REQUIRE(config.num_attention_heads == 16);
    REQUIRE(config.head_dim == 64);
}

TEST_CASE("GptneoxConfig compute_head_dim", "[model][gptneox]")
{
    GptneoxConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.compute_head_dim();
    REQUIRE(config.head_dim == 64);
}

TEST_CASE("GptneoxConfig validate", "[model][gptneox]")
{
    GptneoxConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.head_dim = 64;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("GptneoxConfig validate fails on bad head_dim", "[model][gptneox]")
{
    GptneoxConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.head_dim = 63;
    REQUIRE_THROWS(config.validate());
}

TEST_CASE("GptneoxConfig build_attention_layers default", "[model][gptneox]")
{
    GptneoxConfig config;
    config.num_hidden_layers = 4;
    config.build_attention_layers();
    REQUIRE(config.attention_layers.size() == 4);
    REQUIRE(config.attention_layers[0] == "global");
    REQUIRE(config.attention_layers[1] == "local");
    REQUIRE(config.is_local_attention_layer(1));
    REQUIRE_FALSE(config.is_local_attention_layer(0));
}

TEST_CASE(
    "GptneoxConfig parse attention_types matches HuggingFace expand",
    "[model][gptneox]")
{
    nlohmann::json j = {
        {"num_hidden_layers", 12},
        {"attention_types", {{{"global", "local"}, 6}}}};
    GptneoxConfig config;
    config.num_hidden_layers = 12;
    parse_gptneox_attention_layers(j, config);
    REQUIRE(config.attention_layers.size() == 12);
    for (Index i = 0; i < 12; ++i)
    {
        REQUIRE(
            config.attention_layers[static_cast<std::size_t>(i)] ==
            (i % 2 == 0 ? "global" : "local"));
    }
}


TEST_CASE(
    "GptneoxConfig parse HF default attention_types produces 24 layers",
    "[model][gptneox]")
{
    nlohmann::json j = {
        {"num_hidden_layers", 24},
        {"attention_types", {{{"global", "local"}, 12}}}};
    GptneoxConfig config;
    config.num_hidden_layers = 24;
    config.hidden_size = 2048;
    config.num_attention_heads = 16;
    config.head_dim = 128;
    parse_gptneox_attention_layers(j, config);
    REQUIRE(config.attention_layers.size() == 24);
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE(
    "GptneoxConfig parse attention_layers array round-trip",
    "[model][gptneox]")
{
    nlohmann::json j = {
        {"attention_layers", {"global", "local", "global"}}};
    GptneoxConfig config;
    parse_gptneox_attention_layers(j, config);
    REQUIRE(config.attention_layers.size() == 3);
    REQUIRE(config.attention_layers[2] == "global");
}
