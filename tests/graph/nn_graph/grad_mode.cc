/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/grad_mode.cc
 * Tests for GradMode (no_grad) - ops don't set producer when disabled.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("GradMode disabled: Add does not set producer")
{
    NNGraph g("no_grad_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    NNGraph::TensorNode* z = nullptr;
    {
        GradMode::Guard guard;
        z = add(Scalar(1.0), x, Scalar(1.0), y, "z");
    }

    // With GradMode disabled, add() did not set producer
    REQUIRE(z != nullptr);
    REQUIRE_FALSE(z->has_producer());
    REQUIRE(z->is_leaf());
}

TEST_CASE("GradMode enabled: Add sets producer")
{
    NNGraph g("grad_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(Scalar(1.0), x, Scalar(1.0), y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
}
