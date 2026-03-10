/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/multiply_slice.cc
 * Test TensorGraph multiply_slice operation against nntile::tensor::multiply_slice.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/multiply_slice.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_slice.hh"
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
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_half = 0.5;
constexpr Scalar alpha_two = 2.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Slice shape: dst shape with axis removed
static std::vector<Index> slice_shape(
    const std::vector<Index>& dst_shape,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(dst_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if(i != axis)
        {
            out.push_back(dst_shape[i]);
        }
    }
    return out;
}

template<typename T>
void check_multiply_slice_vs_tensor_api(
    const std::vector<Index>& dst_shape,
    Index axis,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());
    const Index axis_size = dst_shape[axis];

    // --- TensorGraph path ---
    TensorGraph graph("multiply_slice_test");
    auto* src_node = graph.data(src_sh, "src", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::multiply_slice(alpha, src_node, dst_node, axis);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_sh, src_sh);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
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
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::multiply_slice<T>(alpha, src_t, dst_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = dst_t.get_tile(0);
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

TEST_CASE("TensorGraph multiply_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_2}, "src");  // slice for dst [dim_2, dim_4], axis=1
    auto* dst = graph.data({dim_2, dim_4}, "dst");

    gt::multiply_slice(alpha_one, src, dst, axis_1);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY_SLICE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph multiply_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_2}, "src");
    auto* dst = graph.data({dim_2, dim_4}, "dst");

    REQUIRE_THROWS_AS(
        gt::multiply_slice(alpha_one, src, src, axis_1),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_slice matches nntile::tensor::multiply_slice", "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, alpha_half},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, alpha_two});

    check_multiply_slice_vs_tensor_api<nntile::fp32_t>(dst_shape, axis, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_slice tiled matches untiled", "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Index(1), 1.0},
        std::tuple{std::vector<Index>{2, 4}, Index(0), 1.0});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < src_nelems; ++i)
        src_data[i] = static_cast<float>(Y(i + 1));
    for(Index i = 0; i < dst_nelems; ++i)
        dst_data[i] = static_cast<float>(Y(-i - 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("multiply_slice_untiled");
        auto* src_node = graph.data(src_sh, "src", DataType::FP32);
        auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);
        gt::multiply_slice(alpha, src_node, dst_node, axis);
        TensorGraph::Runtime runtime(graph);
        runtime.compile();
        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("dst");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("multiply_slice_tiled");
        auto* src_node = graph.data(src_sh, "src", DataType::FP32);
        auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);
        gt::multiply_slice(alpha, src_node, dst_node, axis);
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
        tiled_result = runtime.get_output<float>("dst");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
