/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sum_slice.cc
 * Test TensorGraph sum_slice operation against nntile::tensor::sum_slice.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sum_slice.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_slice.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

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

//! Output shape for sum_slice: src shape with axis removed
static std::vector<Index> sum_slice_output_shape(
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
void check_sum_slice_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> dst_shape = sum_slice_output_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sum_slice_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    sum_slice(src_node, dst_node, axis, redux, alpha, beta);

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

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    tensor::TensorTraits src_traits(src_shape, src_shape);
    tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    tensor::Tensor<T> src_t(src_traits, src_distr);
    tensor::Tensor<T> dst_t(dst_traits, dst_distr);

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

    tensor::sum_slice<T>(alpha, src_t, beta, dst_t, axis, redux);
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

TEST_CASE("TensorGraph sum_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_4, dim_5}, "src");
    auto* dst = graph.data({dim_4}, "dst");  // axis=1: sum over dim_5

    sum_slice(src, dst, axis_1, redux_none, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().size() == 1);
    REQUIRE(dst->shape()[0] == dim_4);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUM_SLICE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sum_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_4, dim_5}, "src");

    REQUIRE_THROWS_AS(
        sum_slice(src, src, axis_1, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sum_slice matches tensor::sum_slice", "[graph][tensor]")
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

    check_sum_slice_vs_tensor_api<nntile::fp32_t>(
        src_shape, axis, redux, alpha, beta);
}
