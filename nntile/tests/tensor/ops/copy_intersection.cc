/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/copy_intersection.cc
 * Test TensorGraph copy_intersection operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/copy_intersection.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/copy_intersection.hh"
#include "nntile/tensor/ops/scatter.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <memory>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph copy_intersection structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    nntile::TensorRef src = graph.data({dim0, dim1});
    src->set_name("src");
    nntile::TensorRef dst = graph.data({dim0, dim1});
    dst->set_name("dst");
    std::vector<Index> src_offset{0, 0};
    std::vector<Index> dst_offset{0, 0};

    gt::copy_intersection(src, src_offset, dst, dst_offset);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "COPY_INTERSECTION");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE(
    "TensorGraph copy_intersection rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef t = graph.data({5, 4});
    t->set_name("t");
    std::vector<Index> offset{0, 0};

    REQUIRE_THROWS_AS(gt::copy_intersection(nullptr, offset, t, offset),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::copy_intersection(t, offset, nullptr, offset),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, src_off, dst_off] =
        GENERATE(std::tuple{std::vector<Index>{4, 6},
                     std::vector<Index>{0, 0},
                     std::vector<Index>{0, 0}},
            std::tuple{std::vector<Index>{3, 4},
                std::vector<Index>{0, 0},
                std::vector<Index>{0, 0}});

    using T = nntile::fp32_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems, 0.0f);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(i + 1);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("copy_intersection_untiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::copy_intersection(src_node, src_off, dst_node, dst_off);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run: set tiling on every axis group ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("copy_intersection_tiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection INT64 tiled matches untiled",
    "[graph][tensor]")
{
    const std::vector<Index> shape{8, 3};
    const std::vector<Index> src_off{0, 0};
    const std::vector<Index> dst_off{0, 0};
    const Index nelems = 24;

    std::vector<std::int64_t> src_data(nelems);
    std::vector<std::int64_t> dst_data(nelems, 0);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[static_cast<size_t>(i)] = static_cast<std::int64_t>(i + 1);
    }

    std::vector<std::int64_t> untiled_result;
    {
        TensorGraph graph("copy_intersection_int64_untiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::INT64);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::INT64);
    dst_node->set_name("dst");
        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<std::int64_t>(dst_node);
    }

    std::vector<std::int64_t> tiled_result;
    {
        TensorGraph graph("copy_intersection_int64_tiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::INT64);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::INT64);
    dst_node->set_name("dst");
        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<std::int64_t>(dst_node);
    }

    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(tiled_result[i] == untiled_result[i]);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection BOOL tiled matches untiled",
    "[graph][tensor]")
{
    const std::vector<Index> shape{8, 3};
    const std::vector<Index> src_off{0, 0};
    const std::vector<Index> dst_off{0, 0};
    const Index nelems = 24;

    // Avoid std::vector<bool> (no .data()); bind via contiguous bool buffer.
    std::unique_ptr<bool[]> src_data(new bool[static_cast<size_t>(nelems)]);
    std::unique_ptr<bool[]> dst_data(new bool[static_cast<size_t>(nelems)]);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[static_cast<size_t>(i)] = (i % 3) != 0;
        dst_data[static_cast<size_t>(i)] = false;
    }

    std::vector<bool> untiled_result;
    {
        TensorGraph graph("copy_intersection_bool_untiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::BOOL);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::BOOL);
    dst_node->set_name("dst");
        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(
            src_node, src_data.get(), static_cast<size_t>(nelems));
        runtime.bind_data(
            dst_node, dst_data.get(), static_cast<size_t>(nelems));
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<bool>(dst_node);
    }

    for (Index i = 0; i < nelems; ++i)
    {
        dst_data[static_cast<size_t>(i)] = false;
    }

    std::vector<bool> tiled_result;
    {
        TensorGraph graph("copy_intersection_bool_tiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::BOOL);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(shape, DataType::BOOL);
    dst_node->set_name("dst");
        gt::copy_intersection(src_node, src_off, dst_node, dst_off);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(
            src_node, src_data.get(), static_cast<size_t>(nelems));
        runtime.bind_data(
            dst_node, dst_data.get(), static_cast<size_t>(nelems));
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<bool>(dst_node);
    }

    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(tiled_result[i] == untiled_result[i]);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scatter INT64 into tiled logical",
    "[graph][tensor]")
{
    // Mirrors torch_nntile label ingress: single-tile INT64 staging scatters
    // into a multi-tile logical (axes are not merged by scatter).
    const std::vector<Index> shape{8};
    std::vector<std::int64_t> src_data(8);
    for (Index i = 0; i < 8; ++i)
    {
        src_data[static_cast<size_t>(i)] = static_cast<std::int64_t>(i + 1);
    }

    TensorGraph graph("scatter_int64_tiled");
    nntile::TensorRef staging = graph.data(shape, DataType::INT64);
    staging->set_name("staging");
    nntile::TensorRef logical = graph.data(shape, DataType::INT64);
    logical->set_name("logical");
    gt::scatter(staging, logical);
    // Tile only the logical axis group; staging stays single-tile.
    logical->axis(0)->set_tiling(std::vector<Index>{4, 4});

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(staging, src_data);
    runtime.execute();
    runtime.wait();

    const auto out = runtime.get_output<std::int64_t>(logical);
    REQUIRE(out.size() == src_data.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        REQUIRE(out[i] == src_data[i]);
    }
}
