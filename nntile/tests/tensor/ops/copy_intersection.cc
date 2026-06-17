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
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
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

    auto *src = graph.data({dim0, dim1})->set_name("src");
    auto *dst = graph.data({dim0, dim1})->set_name("dst");
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
    auto *t = graph.data({5, 4})->set_name("t");
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
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

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
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

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
