/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sum.cc
 * Test TensorGraph sum operation against nntile::tensor::sum.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sum.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_sum_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sum_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* dst_node = graph.data({}, "dst", DataType::FP32);  // scalar
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::sum(src_node, dst_node, alpha, beta);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(1);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }
    dst_data[0] = (beta != beta_zero) ? 1.0f : 0.0f;

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits dst_traits({}, {});
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(1, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> dst_t(dst_traits, dst_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        loc[0] = static_cast<Y>(dst_data[0]);
        loc.release();
    }

    nntile::tensor::sum<T>(alpha, src_t, beta, dst_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(1);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        tensor_result[0] = static_cast<float>(loc[0]);
        loc.release();
    }

    REQUIRE(graph_result.size() == 1);
    REQUIRE(tensor_result.size() == 1);
    REQUIRE(std::abs(graph_result[0] - tensor_result[0]) < tolerance);
}

TEST_CASE("TensorGraph sum structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");
    auto* dst = graph.data({}, "dst");

    gt::sum(src, dst, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().empty());

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUM");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sum rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* t = graph.data({4, 5}, "t");

    REQUIRE_THROWS_AS(gt::sum(t, t, alpha_one, beta_zero), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sum matches nntile::tensor::sum", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 0.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 0.0, std::vector<Index>{6}},
        std::tuple{1.0, 1.0, std::vector<Index>{3, 4}});

    check_sum_vs_tensor_api<nntile::fp32_t>(shape, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sum tiled matches untiled", "[graph][tensor]")
{
    const auto [alpha, beta, src_shape] = GENERATE(
        std::tuple{1.0, 0.0, std::vector<Index>{4, 6}},
        std::tuple{1.0, 1.0, std::vector<Index>{6}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }
    std::vector<float> dst_data(1);
    dst_data[0] = (beta != beta_zero) ? 1.0f : 0.0f;

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("sum_untiled");
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* dst_node = graph.data({}, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::sum(src_node, dst_node, alpha, beta);

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
        TensorGraph graph("sum_tiled");
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* dst_node = graph.data({}, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::sum(src_node, dst_node, alpha, beta);
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
