/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/relu.cc
 * Test NNGraph relu autograd operation.
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
    "NNGraph relu structure", "[graph][nn_graph]")
{
    const auto shape = GENERATE(
        std::vector<Index>{2, 3},
        std::vector<Index>{4, 5});

    NNGraph g("relu_structure");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = relu(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "RELU");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph relu backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(-1.0)});

    NNGraph g("relu_backward");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = relu(x, "y");

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph relu forward and backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(1.0)},
        std::tuple{std::vector<Index>{6}, Scalar(2.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Scalar(-1.0)});

    NNGraph g("relu");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = relu(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}
