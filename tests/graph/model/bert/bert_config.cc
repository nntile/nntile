/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/bert/bert_config.cc
 * Tests for BertConfig.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/model/bert/bert_config.hh"

using namespace nntile::model::bert;

TEST_CASE("BertConfig default values", "[model][bert]")
{
    BertConfig config;
    REQUIRE(config.vocab_size == 30522);
    REQUIRE(config.hidden_size == 768);
    REQUIRE(config.num_attention_heads == 12);
    REQUIRE(config.head_dim() == 64);
}

TEST_CASE("BertConfig validate", "[model][bert]")
{
    BertConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("BertConfig validate fails on bad hidden_size", "[model][bert]")
{
    BertConfig config;
    config.hidden_size = 511;
    config.num_attention_heads = 8;
    REQUIRE_THROWS(config.validate());
}
