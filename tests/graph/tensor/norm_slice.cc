/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/norm_slice.cc
 * Test TensorGraph norm_slice operation against nntile::tensor::norm_slice.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/norm_slice.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_slice.hh"
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
constexpr int redux_none = 0;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float y_init_overwrite = 0.0f;
constexpr float y_init_accumulate = 1.0f;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;
constexpr Index x_fill_offset = 1;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr Index dim_6 = 6;

} // anonymous namespace

//! Output shape for norm_slice: src shape with axis removed
static std::vector<Index> norm_slice_output_shape(
    const std::vector<Index>& src_shape,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(src_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if(i != axis)
        {
            out.push_back(src_shape[i]);
        }
    }
    return out;
}

template<typename T>
void check_norm_slice_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> dst_shape = norm_slice_output_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path (5-arg: creates distinct output) ---
    TensorGraph graph("norm_slice_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);

    auto* out_node = gt::norm_slice(alpha, src_node, beta, dst_node, "out", axis, redux);
    out_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }

    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? y_init_accumulate : y_init_overwrite;
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("out");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    nntile::tensor::TensorTraits out_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    std::vector<int> out_distr(out_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> src2_t(dst_traits, dst_distr);
    nntile::tensor::Tensor<T> out_t(out_traits, out_distr);

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
        auto tile = src2_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    {
        auto tile = out_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::norm_slice<T>(alpha, src_t, beta, src2_t, out_t, axis, redux);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = out_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
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

TEST_CASE("TensorGraph norm_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_4, dim_5}, "src");
    auto* dst = graph.data({dim_4}, "dst");  // axis=1: norm over dim_5

    auto* out = gt::norm_slice(alpha_one, src, beta_zero, dst, "out", axis_1, redux_none);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape().size() == 1);
    REQUIRE(out->shape()[0] == dim_4);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "NORM_SLICE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE("TensorGraph norm_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_4, dim_5}, "src");
    auto* dst = graph.data({dim_4}, "dst");

    REQUIRE_THROWS_AS(
        gt::norm_slice(alpha_one, src, beta_zero, dst, dst, axis_1, redux_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm_slice matches nntile::tensor::norm_slice", "[graph][tensor]")
{
    const auto [src_shape, axis, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_1, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_3, dim_6}, axis_0, redux_none, alpha_two, beta_zero},
        std::tuple{std::vector<Index>{dim_3, dim_6}, axis_1, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, redux_none, alpha_one, beta_half});

    check_norm_slice_vs_tensor_api<nntile::fp32_t>(
        src_shape, axis, redux, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm_slice tiled matches untiled", "[graph][tensor]")
{
    const auto [src_shape, axis, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, redux_none, alpha_one, beta_zero});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> dst_shape = norm_slice_output_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? y_init_accumulate : y_init_overwrite;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("norm_slice_untiled");
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);

        auto* out_node = gt::norm_slice(alpha, src_node, beta, dst_node, "out", axis, redux);
        out_node->mark_output(true);

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("out");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("norm_slice_tiled");
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);

        auto* out_node = gt::norm_slice(alpha, src_node, beta, dst_node, "out", axis, redux);
        out_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("out");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
