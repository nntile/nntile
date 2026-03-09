/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/add.cc
 * Test TileGraph add operation against nntile::tile::add.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tile/add.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/add.hh"
#include "nntile/tile/tile.hh"

using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;

template<typename T>
void check_tile_add_vs_tile_api(
    const std::vector<Index>& shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    TileGraph graph("tile_add_test");
    auto* x_node = graph.data(shape, "x", DataType::FP32);
    auto* y_node = graph.data(shape, "y", DataType::FP32);
    x_node->mark_input(true);
    y_node->mark_input(true);

    auto* z_node = tg::add(alpha, x_node, beta, y_node, "z");
    z_node->mark_output(true);

    TileGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("z");

    nntile::tile::Tile<T> src1(shape);
    nntile::tile::Tile<T> src2(shape);
    nntile::tile::Tile<T> dst(shape);

    {
        auto loc1 = src1.acquire(STARPU_W);
        auto loc2 = src2.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc1[i] = static_cast<Y>(x_data[i]);
            loc2[i] = static_cast<Y>(y_data[i]);
        }
        loc1.release();
        loc2.release();
    }

    nntile::tile::add<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();

    std::vector<float> tile_result(nelems);
    {
        auto loc = dst.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tile_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tile_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tile_result[i]) < tol);
    }
}

TEST_CASE("TileGraph add structure", "[graph][tile]")
{
    const auto [alpha, beta] = GENERATE(
        std::tuple{1.0, 1.0},
        std::tuple{2.0, 3.0},
        std::tuple{0.5, -1.0});
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TileGraph graph("test");

    auto* x = graph.data({dim0, dim1}, "x");
    auto* y = graph.data({dim0, dim1}, "y");

    auto* z = tg::add(alpha, x, beta, y, "z");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(z->shape()[0] == dim0);
    REQUIRE(z->shape()[1] == dim1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "TILE_ADD");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == z);
}

TEST_CASE("TileGraph add rejects duplicate tiles", "[graph][tile]")
{
    TileGraph graph("test");
    auto* x = graph.data({4, 5}, "x");

    REQUIRE_THROWS_AS(tg::add(1.0, x, 1.0, x, "z"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph add matches nntile::tile::add", "[graph][tile]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 5}},
        std::tuple{0.5, -1.0, std::vector<Index>{6}},
        std::tuple{1.0, 2.0, std::vector<Index>{3, 4}},
        std::tuple{-0.5, 1.5, std::vector<Index>{2, 2}});

    check_tile_add_vs_tile_api<nntile::fp32_t>(shape, alpha, beta);
}
