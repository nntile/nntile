/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/t5/t5_config.cc
 * Tests for T5Config.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/model/t5/t5_config.hh"

using namespace nntile::model::t5;

TEST_CASE("T5Config default values", "[model][t5]")
{
    T5Config config;
    REQUIRE(config.vocab_size == 32100);
    REQUIRE(config.d_model == 512);
    REQUIRE(config.num_heads == 8);
    REQUIRE(config.d_kv == 64);
    REQUIRE(config.head_dim() == 64);
}

TEST_CASE("T5Config head_dim and inner_dim", "[model][t5]")
{
    T5Config config;
    config.d_model = 64;
    config.num_heads = 4;
    config.d_kv = 16;
    REQUIRE(config.head_dim() == 16);
    REQUIRE(config.inner_dim() == 64);
}

TEST_CASE("T5Config validate", "[model][t5]")
{
    T5Config config;
    config.d_model = 64;
    config.num_heads = 4;
    config.d_kv = 16;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE(
    "T5Config validate allows d_kv independent of d_model",
    "[model][t5]")
{
    T5Config config;
    config.d_model = 512;
    config.num_heads = 6;
    config.d_kv = 64;
    REQUIRE(config.inner_dim() == 384);
    REQUIRE(config.inner_dim() != config.d_model);
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("T5Config validate fails on non-positive d_kv", "[model][t5]")
{
    T5Config config;
    config.d_model = 64;
    config.num_heads = 4;
    config.d_kv = 0;
    REQUIRE_THROWS(config.validate());
}
