/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/hypot_inplace.cc
 * Test TensorGraph hypot_inplace operation against nntile::tensor::hypot_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/hypot_inplace.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr Scalar beta_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_hypot_inplace_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("hypot_inplace_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::hypot_inplace(alpha, src_node, beta, dst_node);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems);
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

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(traits, distr);
    nntile::tensor::Tensor<T> dst_t(traits, distr);

    {
        auto tile1 = src_t.get_tile(0);
        auto tile2 = dst_t.get_tile(0);
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

    nntile::tensor::hypot_inplace<T>(alpha, src_t, beta, dst_t);
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

TEST_CASE("TensorGraph hypot_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");
    auto* dst = graph.data({dim0, dim1}, "dst");

    gt::hypot_inplace(alpha_one, src, beta_one, dst);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "HYPOT_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph hypot_inplace rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* t = graph.data({4, 5}, "t");

    REQUIRE_THROWS_AS(
        gt::hypot_inplace(alpha_one, t, beta_one, t),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph hypot_inplace matches nntile::tensor::hypot_inplace", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 5}},
        std::tuple{0.5, 1.0, std::vector<Index>{6}},
        std::tuple{1.0, 2.0, std::vector<Index>{3, 4}});

    check_hypot_inplace_vs_tensor_api<nntile::fp32_t>(shape, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph hypot_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 6}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 6}},
        std::tuple{0.5, -1.0, std::vector<Index>{6}});

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
        TensorGraph graph("hypot_inplace_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::hypot_inplace(alpha, src_node, beta, dst_node);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
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
        TensorGraph graph("hypot_inplace_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::hypot_inplace(alpha, src_node, beta, dst_node);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
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
