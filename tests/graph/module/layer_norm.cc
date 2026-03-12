/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/layer_norm.cc
 * Tests for LayerNorm module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/graph/module/layer_norm.hh"
#include "nntile/graph/tensor/graph.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("LayerNorm ConstructorCreatesParameters", "[module]")
{
    NNGraph g("layer_norm");

    LayerNorm ln(&g, "ln", 64, 0, 1e-5f);
    REQUIRE(ln.gamma_tensor() != nullptr);
    REQUIRE(ln.beta_tensor() != nullptr);
    REQUIRE(ln.gamma_tensor()->shape() == std::vector<Index>({64}));
    REQUIRE(ln.beta_tensor()->shape() == std::vector<Index>({64}));
    REQUIRE(ln.gamma_tensor()->name() == "ln_gamma");
    REQUIRE(ln.beta_tensor()->name() == "ln_beta");
    REQUIRE(ln.parameters().size() == 2);
}

TEST_CASE("LayerNorm Callable", "[module]")
{
    NNGraph g("layer_norm_callable");
    auto* input = g.tensor({4, 64}, "input", DataType::FP32);
    LayerNorm ln(&g, "ln", 64, 1, 1e-5f);
    auto* output = ln.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({4, 64}));
}

TEST_CASE("LayerNorm BuildForward", "[module]")
{
    NNGraph g("layer_norm");

    auto* input = g.tensor({2, 3, 4}, "input", DataType::FP32);
    LayerNorm ln(&g, "ln", 4, 2, 1e-5f);

    auto* output = ln.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 3, 4}));
    REQUIRE(output->name() == "ln_out");
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE("LayerNorm Repr", "[module]")
{
    NNGraph g("layer_norm");
    LayerNorm ln(&g, "ln", 768, 0, 1e-5f);
    std::string r = ln.repr();
    REQUIRE(r.find("LayerNorm") != std::string::npos);
    REQUIRE(r.find("768") != std::string::npos);
}
