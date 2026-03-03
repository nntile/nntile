/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sum_fiber.cc
 * Test TensorGraph sum_fiber operation against nntile::tensor::sum_fiber.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sum_fiber.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_fiber.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Index batch_ndim_none = 0;
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

//! Compute output shape for sum_fiber: y has shape {x_shape[axis]} for batch_ndim=0
static std::vector<Index> sum_fiber_output_shape(
    const std::vector<Index>& x_shape,
    Index axis,
    Index batch_ndim)
{
    std::vector<Index> out_shape;
    out_shape.reserve(batch_ndim + 1);
    out_shape.push_back(x_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[x_shape.size() - batch_ndim + i]);
    }
    return out_shape;
}

template<typename T>
void check_sum_fiber_vs_tensor_api(
    const std::vector<Index>& x_shape,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> y_shape = sum_fiber_output_shape(x_shape, axis, batch_ndim);
    const Index y_nelems = std::accumulate(
        y_shape.begin(), y_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sum_fiber_test");
    auto* x_node = graph.data(x_shape, "x", DataType::FP32);
    auto* y_node = graph.data(y_shape, "y", DataType::FP32);
    x_node->mark_input(true);
    y_node->mark_input(true);
    y_node->mark_output(true);

    sum_fiber(x_node, y_node, axis, batch_ndim, redux, alpha, beta);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }

    std::vector<float> y_data(y_nelems);
    for(Index i = 0; i < y_nelems; ++i)
    {
        y_data[i] = (beta != beta_zero) ? y_init_accumulate : y_init_overwrite;
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("y");

    // --- Direct tensor API path (same input data) ---
    tensor::TensorTraits x_traits(x_shape, x_shape);
    tensor::TensorTraits y_traits(y_shape, y_shape);
    std::vector<int> x_distr(x_traits.grid.nelems, distr_rank_single);
    std::vector<int> y_distr(y_traits.grid.nelems, distr_rank_single);
    tensor::Tensor<T> src(x_traits, x_distr);
    tensor::Tensor<T> dst(y_traits, y_distr);

    {
        auto tile = src.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < x_nelems; ++i)
        {
            loc[i] = static_cast<Y>(x_data[i]);
        }
        loc.release();
    }

    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < y_nelems; ++i)
        {
            loc[i] = static_cast<Y>(y_data[i]);
        }
        loc.release();
    }

    tensor::sum_fiber<T>(alpha, src, beta, dst, axis, batch_ndim, redux);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(y_nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < y_nelems; ++i)
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

TEST_CASE("TensorGraph sum_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* x = graph.data({dim_4, dim_5}, "x");
    auto* y = graph.data({dim_4}, "y");  // axis=0: sum over dim_5, keep dim_4

    sum_fiber(x, y, axis_0, batch_ndim_none, redux_none, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(y->shape().size() == 1);
    REQUIRE(y->shape()[0] == dim_4);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUM_FIBER");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sum_fiber matches tensor::sum_fiber", "[graph][tensor]")
{
    const auto [x_shape, axis, batch_ndim, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_3, dim_6}, axis_0, batch_ndim_none, redux_none, alpha_two, beta_zero},
        std::tuple{std::vector<Index>{dim_3, dim_6}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, batch_ndim_none, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_0, batch_ndim_none, redux_none, alpha_one, beta_half});

    check_sum_fiber_vs_tensor_api<nntile::fp32_t>(
        x_shape, axis, batch_ndim, redux, alpha, beta);
}
