/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/rms_norm.cc
 * Tests for RMSNorm module.
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
#include "nntile/graph/module/rms_norm.hh"
#include "nntile/graph/tensor/graph.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("RMSNorm ConstructorCreatesParameters", "[module]")
{
    NNGraph g("rms_norm");

    RMSNorm rn(&g, "rn", 64, 0, 1e-6f);
    REQUIRE(rn.gamma_tensor() != nullptr);
    REQUIRE(rn.gamma_tensor()->shape() == std::vector<Index>({64}));
    REQUIRE(rn.gamma_tensor()->name() == "rn_gamma");
    REQUIRE(rn.parameters().size() == 1);
}

TEST_CASE("RMSNorm Callable", "[module]")
{
    NNGraph g("rms_norm_callable");
    auto* input = g.tensor({4, 64}, "input", DataType::FP32);
    RMSNorm rn(&g, "rn", 64, 1, 1e-6f);
    auto* output = rn.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({4, 64}));
}

TEST_CASE("RMSNorm BuildForward", "[module]")
{
    NNGraph g("rms_norm");

    auto* input = g.tensor({2, 3, 4}, "input", DataType::FP32);
    RMSNorm rn(&g, "rn", 4, 2, 1e-6f);

    auto* output = rn.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 3, 4}));
    REQUIRE(output->name() == "rn_out");
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE("RMSNorm Repr", "[module]")
{
    NNGraph g("rms_norm");
    RMSNorm rn(&g, "rn", 768, 0, 1e-6f);
    std::string r = rn.repr();
    REQUIRE(r.find("RMSNorm") != std::string::npos);
    REQUIRE(r.find("768") != std::string::npos);
}
