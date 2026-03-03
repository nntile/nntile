/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn/grad_mode.cc
 * Tests for GradMode (no_grad) - ops don't set producer when disabled.
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
    "GradMode disabled: Add does not set producer")
{
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));

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
    "GradMode enabled: Add sets producer")
{
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));

    NNGraph g("grad_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
}
