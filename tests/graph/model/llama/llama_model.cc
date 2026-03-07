/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/llama/llama_model.cc
 * Tests for LlamaModel.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/model/llama/llama_model.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;

TEST_CASE("LlamaModel forward builds output", "[model][llama]")
{
    NNGraph g("llama_model");
    LlamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();

    auto* input_ids = g.tensor({4, 2}, "input_ids", DataType::INT64);
    LlamaModel model(&g, "model", config);
    auto* output = model.forward(input_ids);

    REQUIRE(output != nullptr);
    // Output: (hidden, seq, batch) = (8, 4, 2)
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
}
