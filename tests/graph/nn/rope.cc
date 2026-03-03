/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/rope.cc
 * Test NNGraph rope autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

// RoPE requires src.shape[0] == 2*sin.shape[0]
static std::vector<Index> make_src_shape(const std::vector<Index>& sin_shape)
{
    std::vector<Index> src_shape = {sin_shape[0] * 2};
    src_shape.insert(src_shape.end(), sin_shape.begin() + 1, sin_shape.end());
    return src_shape;
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope structure", "[graph][nn_graph]")
{
    const auto sin_shape = GENERATE(
        std::vector<Index>{2, 4},
        std::vector<Index>{4, 3, 2});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope_structure");
    auto* sin = g.tensor(sin_shape, "sin", DataType::FP32, false);
    auto* cos = g.tensor(sin_shape, "cos", DataType::FP32, false);
    auto* x = g.tensor(src_shape, "x", DataType::FP32);
    auto* y = rope(sin, cos, x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == src_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope backward", "[graph][nn_graph]")
{
    const auto [sin_shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 3}, Scalar(-1.0)});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope_backward");
    auto* sin = g.tensor(sin_shape, "sin", DataType::FP32, false);
    auto* cos = g.tensor(sin_shape, "cos", DataType::FP32, false);
    auto* x = g.tensor(src_shape, "x", DataType::FP32);
    auto* y = rope(sin, cos, x, "y");

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == src_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope forward and backward", "[graph][nn_graph]")
{
    const auto [sin_shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Scalar(-1.0)});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope");
    auto* sin = g.tensor(sin_shape, "sin", DataType::FP32, false);
    auto* cos = g.tensor(sin_shape, "cos", DataType::FP32, false);
    auto* x = g.tensor(src_shape, "x", DataType::FP32, true);
    auto* y = rope(sin, cos, x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}
