/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/norm_fiber_inplace.cc
 * Test TensorGraph norm_fiber_inplace operation against nntile::tensor::norm_fiber_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/norm_fiber_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_fiber_inplace.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Index batch_ndim_none = 0;
constexpr int redux_none = 0;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;
constexpr Index x_fill_offset = 1;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr Index dim_6 = 6;

} // anonymous namespace

//! Fiber shape: {tensor_shape[axis]} for batch_ndim=0
static std::vector<Index> fiber_shape(
    const std::vector<Index>& tensor_shape,
    Index axis,
    Index batch_ndim)
{
    std::vector<Index> out;
    out.reserve(batch_ndim + 1);
    out.push_back(tensor_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(tensor_shape[tensor_shape.size() - batch_ndim + i]);
    }
    return out;
}

template<typename T>
void check_norm_fiber_inplace_vs_tensor_api(
    const std::vector<Index>& tensor_shape,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index tensor_nelems = std::accumulate(
        tensor_shape.begin(), tensor_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> fiber_sh = fiber_shape(tensor_shape, axis, batch_ndim);
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("norm_fiber_inplace_test");
    auto* src_node = graph.data(tensor_shape, "src", DataType::FP32);
    auto* dst_node = graph.data(fiber_sh, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::norm_fiber_inplace(alpha, src_node, beta, dst_node, axis, batch_ndim, redux);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> src_data(tensor_nelems);
    std::vector<float> dst_data(fiber_nelems);
    for(Index i = 0; i < tensor_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }
    for(Index i = 0; i < fiber_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? static_cast<float>(Y(i + 10)) : 0.0f;
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(tensor_shape, tensor_shape);
    nntile::tensor::TensorTraits dst_traits(fiber_sh, fiber_sh);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> dst_t(dst_traits, dst_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < tensor_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < fiber_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::norm_fiber_inplace<T>(alpha, src_t, beta, dst_t, axis, batch_ndim, redux);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(fiber_nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < fiber_nelems; ++i)
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

TEST_CASE("TensorGraph norm_fiber_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_2, dim_4}, "src");
    auto* dst = graph.data({dim_4}, "dst");  // axis=1: norm over dim_2

    gt::norm_fiber_inplace(alpha_one, src, beta_one, dst, axis_1, batch_ndim_none, redux_none);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "NORM_FIBER_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph norm_fiber_inplace rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_2, dim_4}, "src");

    REQUIRE_THROWS_AS(
        gt::norm_fiber_inplace(alpha_one, src, beta_one, src, axis_1, batch_ndim_none, redux_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm_fiber_inplace matches nntile::tensor::norm_fiber_inplace", "[graph][tensor]")
{
    const auto [tensor_shape, axis, batch_ndim, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, batch_ndim_none, redux_none, alpha_two, beta_half},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, batch_ndim_none, redux_none, alpha_one, beta_one});

    check_norm_fiber_inplace_vs_tensor_api<nntile::fp32_t>(
        tensor_shape, axis, batch_ndim, redux, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm_fiber_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [tensor_shape, axis, batch_ndim, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_zero});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index tensor_nelems = std::accumulate(
        tensor_shape.begin(), tensor_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> fiber_sh = fiber_shape(tensor_shape, axis, batch_ndim);
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(tensor_nelems);
    for(Index i = 0; i < tensor_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }
    std::vector<float> dst_data(fiber_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? static_cast<float>(Y(i + 10)) : 0.0f;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("norm_fiber_inplace_untiled");
        auto* src_node = graph.data(tensor_shape, "src", DataType::FP32);
        auto* dst_node = graph.data(fiber_sh, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::norm_fiber_inplace(alpha, src_node, beta, dst_node, axis, batch_ndim, redux);

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
        TensorGraph graph("norm_fiber_inplace_tiled");
        auto* src_node = graph.data(tensor_shape, "src", DataType::FP32);
        auto* dst_node = graph.data(fiber_sh, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::norm_fiber_inplace(alpha, src_node, beta, dst_node, axis, batch_ndim, redux);
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
