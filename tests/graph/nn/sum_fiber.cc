/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/sum_fiber.cc
 * Test NNGraph sum_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Index batch_ndim_none = 0;
constexpr int redux_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber structure", "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, Index(1), Scalar(1.0), Scalar(0.0)},
        std::tuple{std::vector<Index>{dim_2, dim_4}, Index(0), Scalar(2.0), Scalar(0.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = sum_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha, beta);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == y_shape);
    REQUIRE(g.num_ops() >= 1);
    bool has_sum_fiber = false;
    for(const auto& op : g.tensor_graph().ops())
    {
        if(op->op_name() == "SUM_FIBER")
        {
            has_sum_fiber = true;
            break;
        }
    }
    REQUIRE(has_sum_fiber);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber backward", "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha, beta, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, Index(1), Scalar(1.0), Scalar(0.0),
                   Scalar(1.0)},
        std::tuple{std::vector<Index>{dim_2, dim_4}, Index(0), Scalar(1.0), Scalar(0.0),
                   Scalar(-1.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber_backward");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = sum_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha, beta);

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber forward and backward", "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha, beta, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Index(1), Scalar(1.0), Scalar(0.0),
                   Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 4}, Index(0), Scalar(1.0), Scalar(0.0),
                   Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 6}, Index(0), Scalar(2.0), Scalar(0.0),
                   Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 6}, Index(1), Scalar(1.0), Scalar(0.5),
                   Scalar(-1.0)},
        std::tuple{std::vector<Index>{2, 3, 4}, Index(2), Scalar(1.0), Scalar(0.0),
                   Scalar(1.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* y = sum_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha, beta);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == y_shape);

    auto* y_grad = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}
