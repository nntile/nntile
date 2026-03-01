/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/add_fiber.cc
 * Tests for NNGraph add_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd AddFiber ForwardAndBackward", "[graph][nn_graph]")
{
    NNGraph g("add_fiber");
    auto* fiber = g.tensor({4}, "fiber", DataType::FP32, true);
    auto* tensor = g.tensor({2, 4}, "tensor", DataType::FP32, true);
    auto* out = add_fiber(1.0, fiber, 1.0, tensor, "out", 1, 0);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == (std::vector<Index>{2, 4}));

    auto* out_grad = g.get_or_create_grad(out, "out_grad");
    fill(Scalar(1.0), out_grad->data());
    out->backward();

    REQUIRE(fiber->has_grad());
    REQUIRE(tensor->has_grad());
    REQUIRE(fiber->grad()->shape() == (std::vector<Index>{4}));
    REQUIRE(tensor->grad()->shape() == (std::vector<Index>{2, 4}));
}
