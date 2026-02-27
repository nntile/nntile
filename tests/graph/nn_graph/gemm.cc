/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/gemm.cc
 * Tests for NNGraph gemm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd Gemm ForwardAndBackward", "[graph][nn_graph]")
{
    NNGraph g("gemm");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32, true);
    auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c");

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{2, 4}));

    auto* c_grad = g.get_or_create_grad(c, "c_grad");
    fill(Scalar(1.0), c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{2, 3}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
}
