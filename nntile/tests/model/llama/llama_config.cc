/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/llama/llama_config.cc
 * Tests for LlamaConfig.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/model/llama/llama_config.hh"

using namespace nntile::model::llama;

TEST_CASE("LlamaConfig default values", "[model][llama]")
{
    LlamaConfig config;
    REQUIRE(config.vocab_size == 32000);
    REQUIRE(config.hidden_size == 4096);
    REQUIRE(config.num_attention_heads == 32);
    REQUIRE(config.head_dim == 128);
}

TEST_CASE("LlamaConfig compute_head_dim", "[model][llama]")
{
    LlamaConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.compute_head_dim();
    REQUIRE(config.head_dim == 64);
}

TEST_CASE("LlamaConfig validate", "[model][llama]")
{
    LlamaConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 8;
    config.compute_head_dim();
    REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("LlamaConfig validate fails on bad hidden_size", "[model][llama]")
{
    LlamaConfig config;
    config.hidden_size = 511;
    config.num_attention_heads = 8;
    config.compute_head_dim();
    REQUIRE_THROWS(config.validate());
}
