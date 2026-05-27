/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/gpt2/gpt2_config.cc
 * Tests for Gpt2Config.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/model/gpt2/gpt2_config.hh"

using namespace nntile::graph::model::gpt2;

TEST_CASE("Gpt2Config default values", "[model][gpt2]")
{
    Gpt2Config config;
    REQUIRE(config.vocab_size == 50257);
    REQUIRE(config.hidden_size == 768);
    REQUIRE(config.num_attention_heads == 12);
    REQUIRE(config.head_dim() == 64);
}

TEST_CASE("Gpt2Config head_dim", "[model][gpt2]")
{
    Gpt2Config config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    REQUIRE(config.head_dim() == 64);
}

TEST_CASE("Gpt2Config validate", "[model][gpt2]")
{
    Gpt2Config config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("Gpt2Config validate fails on bad hidden_size", "[model][gpt2]")
{
    Gpt2Config config;
    config.hidden_size = 511;
    config.num_attention_heads = 8;
    REQUIRE_THROWS(config.validate());
}
