/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/fill.cc
 * Test TensorGraph fill operation against nntile::tensor::fill.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/fill.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/fill.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar fill_val = 3.14;

} // anonymous namespace

template<typename T>
void check_fill_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar val)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("fill_test");
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::fill(val, dst_node);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    // Bind with arbitrary initial data (will be overwritten by fill)
    std::vector<float> init_data(nelems, 0.0f);
    runtime.bind_data("dst", init_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> dst(traits, distr);

    nntile::tensor::fill<T>(val, dst);
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

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph fill structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");

    gt::fill(fill_val, src);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "FILL");
    REQUIRE(ops[0]->inputs().size() == 0);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == src);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph fill matches nntile::tensor::fill", "[graph][tensor]")
{
    const auto [val, shape] = GENERATE(
        std::tuple{1.0, std::vector<Index>{4, 5}},
        std::tuple{-2.5, std::vector<Index>{6}},
        std::tuple{0.0, std::vector<Index>{2, 3}},
        std::tuple{3.14, std::vector<Index>{1, 10}});

    check_fill_vs_tensor_api<nntile::fp32_t>(shape, val);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph fill tiled matches untiled", "[graph][tensor]")
{
    const auto [val, shape] = GENERATE(
        std::tuple{1.0, std::vector<Index>{4, 6}},
        std::tuple{-2.5, std::vector<Index>{6}},
        std::tuple{3.14, std::vector<Index>{2, 4}});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("fill_untiled");
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        dst_node->mark_output(true);

        gt::fill(val, dst_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("fill_tiled");
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        dst_node->mark_output(true);

        gt::fill(val, dst_node);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

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
