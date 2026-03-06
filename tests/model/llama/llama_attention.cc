/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/llama/llama_attention.cc
 * Tests for LlamaAttention (sdpa_eager-based).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"
#include "nntile/model/llama/llama_attention.hh"
#include "nntile/model/llama/llama_config.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    NNGraph g("llama_attn");
    LlamaConfig config;
    config.hidden_size = 8;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();

    // Input: (hidden, seq, batch) = (8, 4, 2)
    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}
