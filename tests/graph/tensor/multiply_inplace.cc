/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/multiply_inplace.cc
 * Test TensorGraph multiply_inplace operation against nntile::tensor::multiply_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/multiply_inplace.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/multiply_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha = 2.0;

} // anonymous namespace

template<typename T>
void check_multiply_inplace_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("multiply_inplace_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::multiply_inplace(alpha, src_node, dst_node);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> src_data(nelems), dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
        dst_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> src(traits, distr);
    nntile::tensor::Tensor<T> dst(traits, distr);

    {
        auto tile1 = src.get_tile(0);
        auto tile2 = dst.get_tile(0);
        auto loc1 = tile1.acquire(STARPU_W);
        auto loc2 = tile2.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc1[i] = static_cast<Y>(src_data[i]);
            loc2[i] = static_cast<Y>(dst_data[i]);
        }
        loc1.release();
        loc2.release();
    }

    nntile::tensor::multiply_inplace<T>(alpha, src, dst);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph multiply_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");
    auto* dst = graph.data({dim0, dim1}, "dst");

    gt::multiply_inplace(alpha, src, dst);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph multiply_inplace rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({4, 5}, "src");

    REQUIRE_THROWS_AS(gt::multiply_inplace(alpha, src, src), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_inplace matches nntile::tensor::multiply_inplace", "[graph][tensor]")
{
    const auto [alpha, shape] = GENERATE(
        std::tuple{1.0, std::vector<Index>{4, 5}},
        std::tuple{2.5, std::vector<Index>{4, 5}},
        std::tuple{0.5, std::vector<Index>{2, 3}},
        std::tuple{3.0, std::vector<Index>{6}});

    check_multiply_inplace_vs_tensor_api<nntile::fp32_t>(shape, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [alpha, shape] = GENERATE(
        std::tuple{1.0, std::vector<Index>{4, 6}},
        std::tuple{2.5, std::vector<Index>{2, 4}},
        std::tuple{0.5, std::vector<Index>{6}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems), dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
        dst_data[i] = static_cast<float>(Y(-i - 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("multiply_inplace_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::multiply_inplace(alpha, src_node, dst_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("multiply_inplace_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::multiply_inplace(alpha, src_node, dst_node);
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
