/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/norm.cc
 * Test TensorGraph norm operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/norm.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/norm.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph norm structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *x = graph.data({dim0, dim1})->set_name("x");
    auto *y = graph.data({})->set_name("y");

    gt::norm(x, y, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(y->shape().empty());

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "NORM");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE("TensorGraph norm rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *t = graph.data({5, 4})->set_name("t");

    REQUIRE_THROWS_AS(
        gt::norm(t, t, alpha_one, beta_zero), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm tiled matches untiled",
    "[graph][tensor]")
{
    const auto [alpha, beta, x_shape] =
        GENERATE(std::tuple{1.0, 0.0, std::vector<Index>{4, 6}},
            std::tuple{1.0, 1.0, std::vector<Index>{6}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(x_nelems);
    for (Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
    }
    std::vector<float> y_data(1);
    y_data[0] = (beta != beta_zero) ? 1.0f : 0.0f;

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("norm_untiled");
        auto *x_node = graph.data(x_shape, DataType::FP32)->set_name("x");
        auto *y_node = graph.data({}, DataType::FP32)->set_name("y");
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::norm(x_node, y_node, alpha, beta);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(y_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("norm_tiled");
        auto *x_node = graph.data(x_shape, DataType::FP32)->set_name("x");
        auto *y_node = graph.data({}, DataType::FP32)->set_name("y");
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::norm(x_node, y_node, alpha, beta);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(y_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
