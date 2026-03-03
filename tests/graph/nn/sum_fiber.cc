/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn/sum_fiber.cc
 * Tests for NNGraph sum_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd SumFiber ForwardAndBackward", "[graph][nn_graph]")
{
    const Index axis = GENERATE(Index(1));
    const Index batch_ndim = GENERATE(Index(0));
    const int redux = GENERATE(0);
    const Scalar alpha = GENERATE(Scalar(1.0));
    const Scalar beta = GENERATE(Scalar(0.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("sum_fiber");
    auto* x = g.tensor({2, 4}, "x", DataType::FP32, true);
    auto* y = sum_fiber(x, "y", axis, batch_ndim, redux, alpha, beta);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == (std::vector<Index>{4}));

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == (std::vector<Index>{2, 4}));
}
