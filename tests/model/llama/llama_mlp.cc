/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/llama/llama_mlp.cc
 * Tests for LlamaMLP.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"
#include "nntile/model/llama/llama_config.hh"
#include "nntile/model/llama/llama_mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;

TEST_CASE("LlamaMLP forward builds output", "[model][llama]")
{
    NNGraph g("llama_mlp");
    LlamaConfig config;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.compute_head_dim();

    // Input: (hidden, seq, batch) = (8, 4, 2)
    auto* input = g.tensor({8, 4, 2}, "input", DataType::FP32);
    LlamaMLP mlp(&g, "mlp", config);
    auto* output = mlp.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({8, 4, 2}));
    REQUIRE(mlp.parameters_recursive().size() == 3);
}
