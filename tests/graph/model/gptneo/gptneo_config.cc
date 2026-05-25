/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneo/gptneo_config.cc
 * Tests for GptneoConfig.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/model/gptneo/gptneo_config.hh"

using namespace nntile::model::gptneo;

TEST_CASE("GptneoConfig default values", "[model][gptneo]")
{
    GptneoConfig config;
    REQUIRE(config.vocab_size == 50257);
    REQUIRE(config.hidden_size == 2048);
    REQUIRE(config.num_attention_heads == 16);
    REQUIRE(config.head_dim == 128);
}

TEST_CASE("GptneoConfig compute_head_dim", "[model][gptneo]")
{
    GptneoConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.compute_head_dim();
    REQUIRE(config.head_dim == 64);
}

TEST_CASE("GptneoConfig validate", "[model][gptneo]")
{
    GptneoConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.head_dim = 64;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("GptneoConfig validate fails on bad head_dim", "[model][gptneo]")
{
    GptneoConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.head_dim = 63;
    REQUIRE_THROWS(config.validate());
}
