/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/grad_mode.cc
 * Test NNGraph GradMode (no_grad) - ops don't set producer when disabled.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph grad_mode disabled: add does not set producer", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    NNGraph g("no_grad_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    NNGraph::TensorNode* z = nullptr;
    {
        auto guard = g.no_grad();
        z = add(add_alpha, x, add_beta, y, "z");
    }

    // With grad disabled, add() did not set producer
    REQUIRE(z != nullptr);
    REQUIRE_FALSE(z->has_producer());
    REQUIRE(z->is_leaf());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph grad_mode enabled: add sets producer", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0)});

    NNGraph g("grad_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
}
