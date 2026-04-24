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
#include "nntile/graph/tile.hh"
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
    const std::vector<Index>& src_shape,
    const std::vector<Index>& dst_shape,
    const std::vector<Index>& src_offset,
    const std::vector<Index>& dst_offset,
    const std::vector<Index>* basetile_src_ptr = nullptr,
    const std::vector<Index>* basetile_dst_ptr = nullptr)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());
    const std::vector<Index> basetile_src = basetile_src_ptr
        ? *basetile_src_ptr
        : std::vector<Index>(src_shape.begin(), src_shape.end());
    const std::vector<Index> basetile_dst = basetile_dst_ptr
        ? *basetile_dst_ptr
        : std::vector<Index>(dst_shape.begin(), dst_shape.end());

    // --- TensorGraph path ---
    TensorGraph graph("copy_intersection_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    if(basetile_src_ptr != nullptr)
    {
        for(int k = 0; k < src_node->ndim(); ++k)
        {
            src_node->axis(k)->set_tiling(
                basetile_src[static_cast<size_t>(k)]);
        }
    }
    if(basetile_dst_ptr != nullptr)
    {
        for(int k = 0; k < dst_node->ndim(); ++k)
        {
            dst_node->axis(k)->set_tiling(
                basetile_dst[static_cast<size_t>(k)]);
        }
    }

    gt::copy_intersection(src_node, src_offset, dst_node, dst_offset);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(dst_nelems, 0.0f);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Reference: single-tile nntile::tensor with same values as src_data
    nntile::tensor::TensorTraits ref_src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits ref_dst_traits(dst_shape, dst_shape);
    std::vector<int> distr1(1, distr_rank_single);
    nntile::tensor::Tensor<T> ref_src(ref_src_traits, distr1);
    nntile::tensor::Tensor<T> ref_dst(ref_dst_traits, distr1);
    {
        auto tile = ref_src.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[static_cast<size_t>(i)] = static_cast<Y>(src_data[static_cast<size_t>(i)]);
        }
        loc.release();
    }
    {
        auto tile = ref_dst.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[static_cast<size_t>(i)] = static_cast<Y>(0);
        }
        loc.release();
    }
    nntile::tensor::copy_intersection<T>(ref_src, src_offset, ref_dst, dst_offset);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(static_cast<size_t>(dst_nelems));
    {
        auto tile = ref_dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            tensor_result[static_cast<size_t>(i)] = static_cast<float>(loc[static_cast<size_t>(i)]);
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
        shape, shape, src_off, dst_off, nullptr, nullptr);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection non-zero dst offset",
    "[graph][tensor]")
{
    // Sub-box of 5x6 in a 5x6 dst, shifted
    const std::vector<Index> src_shape{3, 4};
    const std::vector<Index> dst_shape{5, 6};
    const std::vector<Index> src_offset{0, 0};
    const std::vector<Index> dst_offset{1, 1};
    check_copy_intersection_vs_tensor_api<nntile::fp32_t>(src_shape, dst_shape, src_offset,
        dst_offset, nullptr, nullptr);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection different tensor shapes (concat block)",
    "[graph][tensor]")
{
    // Copy all of 3+4 = 7 along axis0 into a larger output starting at 2
    const std::vector<Index> src_shape{3, 4};
    const std::vector<Index> dst_shape{9, 4};
    const std::vector<Index> src_offset{0, 0};
    const std::vector<Index> dst_offset{2, 0};
    check_copy_intersection_vs_tensor_api<nntile::fp32_t>(src_shape, dst_shape, src_offset,
        dst_offset, nullptr, nullptr);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph copy_intersection unequal tilings (same global layout)",
    "[graph][tensor]")
{
    // Same 8x4 logical tensor, different per-axis basetile on src and dst
    const std::vector<Index> shape{8, 4};
    const std::vector<Index> bsrc{2, 2};
    const std::vector<Index> bdst{4, 2};
    const std::vector<Index> off{0, 0};
    check_copy_intersection_vs_tensor_api<nntile::fp32_t>(shape, shape, off, off, &bsrc, &bdst);
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
