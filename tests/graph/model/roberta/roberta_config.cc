/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/roberta/roberta_config.cc
 * Tests for RobertaConfig.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/model/roberta/roberta_config.hh"

using namespace nntile::model::roberta;
using namespace nntile::graph::module;

TEST_CASE("RobertaConfig default values", "[model][roberta]")
{
    RobertaConfig config;
    REQUIRE(config.vocab_size == 50265);
    REQUIRE(config.hidden_size == 768);
    REQUIRE(config.num_attention_heads == 12);
    REQUIRE(config.head_dim() == 64);
    REQUIRE(config.pad_token_id == 1);
}

TEST_CASE("RobertaConfig validate", "[model][roberta]")
{
    RobertaConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("RobertaConfig validate fails on bad hidden_size", "[model][roberta]")
{
    RobertaConfig config;
    config.hidden_size = 511;
    config.num_attention_heads = 8;
    REQUIRE_THROWS(config.validate());
}

TEST_CASE("RobertaConfig hidden_act mapping", "[model][roberta]")
{
    RobertaConfig config;
    REQUIRE(config.hidden_act == "gelu");
    REQUIRE(activation_type_from_config(config) == ActivationType::GELU);

    config.hidden_act = "relu";
    REQUIRE(activation_type_from_config(config) == ActivationType::RELU);

    config.hidden_act = "unsupported_fn";
    REQUIRE_THROWS(config.validate());
}

TEST_CASE("RobertaConfig to_bert_config", "[model][roberta]")
{
    RobertaConfig roberta;
    roberta.hidden_size = 128;
    roberta.num_attention_heads = 4;
    roberta.validate();
    auto bert = to_bert_config(roberta);
    REQUIRE(bert.hidden_size == 128);
    REQUIRE(bert.num_attention_heads == 4);
}
