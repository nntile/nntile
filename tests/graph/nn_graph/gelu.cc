/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/gelu.cc
 * Tests for NNGraph gelu autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd Gelu ForwardAndBackward", "[graph][nn_graph]")
{
    NNGraph g("gelu");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, true);
    auto* y = gelu(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(Scalar(1.0), y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}
