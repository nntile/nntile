/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/softmax.cc
 * Test NNGraph softmax autograd operation.
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
    "NNGraph softmax structure", "[graph][nn_graph]")
{
    const auto [shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1)},
        std::tuple{std::vector<Index>{2, 3, 4}, Index(1)});

    NNGraph g("softmax_structure");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = softmax(x, "y", axis);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == shape);
    REQUIRE(g.num_ops() > 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph softmax backward", "[graph][nn_graph]")
{
    const auto [shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0), Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1), Scalar(-1.0)});

    NNGraph g("softmax_backward");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = softmax(x, "y", axis);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph softmax forward and backward", "[graph][nn_graph]")
{
    const auto [shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0), Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1), Scalar(1.0)},
        std::tuple{std::vector<Index>{6}, Index(0), Scalar(2.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Index(1), Scalar(-1.0)});

    NNGraph g("softmax");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = softmax(x, "y", axis);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}
