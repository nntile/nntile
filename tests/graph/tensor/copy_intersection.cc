/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/copy_intersection.cc
 * Test TensorGraph copy_intersection operation against nntile::tensor::copy_intersection.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/copy_intersection.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/copy_intersection.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_copy_intersection_vs_tensor_api(
    const std::vector<Index>& shape,
    const std::vector<Index>& src_offset,
    const std::vector<Index>& dst_offset)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("copy_intersection_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::copy_intersection(src_node, src_offset, dst_node, dst_offset);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems, 0.0f);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(traits, distr);
    nntile::tensor::Tensor<T> dst_t(traits, distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::copy_intersection<T>(src_t, src_offset, dst_t, dst_offset);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph copy_intersection structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");
    auto* dst = graph.data({dim0, dim1}, "dst");
    std::vector<Index> src_offset{0, 0};
    std::vector<Index> dst_offset{0, 0};

    gt::copy_intersection(src, src_offset, dst, dst_offset);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "COPY_INTERSECTION");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph copy_intersection rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* t = graph.data({4, 5}, "t");
    std::vector<Index> offset{0, 0};

    REQUIRE_THROWS_AS(
        gt::copy_intersection(nullptr, offset, t, offset),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::copy_intersection(t, offset, nullptr, offset),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection matches nntile::tensor::copy_intersection",
    "[graph][tensor]")
{
    const auto [shape, src_off, dst_off] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{0, 0},
                   std::vector<Index>{0, 0}},
        std::tuple{std::vector<Index>{6}, std::vector<Index>{0},
                   std::vector<Index>{0}},
        std::tuple{std::vector<Index>{3, 4}, std::vector<Index>{0, 0},
                   std::vector<Index>{0, 0}});

    check_copy_intersection_vs_tensor_api<nntile::fp32_t>(
        shape, src_off, dst_off);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection tiled matches untiled", "[graph][tensor]")
{
    const auto [shape, src_off, dst_off] = GENERATE(
        std::tuple{std::vector<Index>{4, 6}, std::vector<Index>{0, 0},
                   std::vector<Index>{0, 0}},
        std::tuple{std::vector<Index>{3, 4}, std::vector<Index>{0, 0},
                   std::vector<Index>{0, 0}});

    using T = nntile::fp32_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems, 0.0f);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(i + 1);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("copy_intersection_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::copy_intersection(src_node, src_off, dst_node, dst_off);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run: set tiling on every axis group ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("copy_intersection_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("dst");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
