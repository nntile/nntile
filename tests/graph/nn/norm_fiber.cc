/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/norm_fiber.cc
 * Test NNGraph norm_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index batch_ndim_none = 0;
constexpr int redux_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_fiber structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> out_shape = {x_shape[axis]};

    NNGraph g("norm_fiber_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = norm_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == out_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_fiber forward", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    NNGraph g("norm_fiber_forward");
    auto* x = g.tensor({6, 7}, "x", DataType::FP32, false);
    auto* y = norm_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha);

    x->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    x->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
    if(axis == 0)
        y->data()->axis(0)->set_tiling(std::vector<Index>{2, 4});
    else
        y->data()->axis(0)->set_tiling(std::vector<Index>{3, 4});

    x->mark_input(true);
    y->mark_output(true);

    std::vector<float> x_data(6 * 7);
    for(Index i = 0; i < 6 * 7; ++i)
        x_data[i] = static_cast<float>(i + 1);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> out = runtime.get_output<float>("y");
    const Index expected_size = (axis == 0) ? Index(6) : Index(7);
    REQUIRE(out.size() == static_cast<size_t>(expected_size));
    for(float v : out)
        REQUIRE(v > 0.0);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_fiber backward throws", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Index(1), Scalar(0.5)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    NNGraph g("norm_fiber_backward_throws");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* y = norm_fiber(x, "y", axis, batch_ndim_none, redux_none, alpha);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());

    REQUIRE_THROWS_AS(y->backward(), std::runtime_error);
}
