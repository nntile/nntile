/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sumprod_fiber.cc
 * Test TensorGraph sumprod_fiber operation against nntile::tensor::sumprod_fiber.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sumprod_fiber.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sumprod_fiber.hh"
#include "nntile/tensor/tensor.hh"

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
constexpr Scalar alpha_half = 0.5;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Dst shape for sumprod_fiber: {src_shape[axis]} (1D)
static std::vector<Index> sumprod_fiber_dst_shape(
    const std::vector<Index>& src_shape,
    Index axis)
{
    return {src_shape[axis]};
}

template<typename T>
void check_sumprod_fiber_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> dst_shape = sumprod_fiber_dst_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sumprod_fiber_test");
    auto* src1_node = graph.data(src_shape, "src1", DataType::FP32);
    auto* src2_node = graph.data(src_shape, "src2", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src1_node->mark_input(true);
    src2_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::sumprod_fiber(src1_node, src2_node, dst_node, axis, redux, alpha, beta);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src1_data(src_nelems);
    std::vector<float> src2_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src1_data[i] = static_cast<float>(Y((i + 1) * (i + 2)));
        src2_data[i] = static_cast<float>(Y(1.0 / (i + 1)));
    }
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? 1.0f : 0.0f;
    }

    runtime.bind_data("src1", src1_data);
    runtime.bind_data("src2", src2_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src1_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> src2_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> dst_t(dst_traits, dst_distr);

    {
        auto tile = src1_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src1_data[i]);
        }
        loc.release();
    }
    {
        auto tile = src2_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src2_data[i]);
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

    nntile::tensor::sumprod_fiber<T>(alpha, src1_t, src2_t, beta, dst_t, axis, redux);
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

TEST_CASE("TensorGraph sumprod_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src1 = graph.data({dim_2, dim_4}, "src1");
    auto* src2 = graph.data({dim_2, dim_4}, "src2");
    auto* dst = graph.data({dim_4}, "dst");  // axis=1: fiber length dim_4

    gt::sumprod_fiber(src1, src2, dst, axis_1, redux_none, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUMPROD_FIBER");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sumprod_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src1 = graph.data({dim_2, dim_4}, "src1");
    auto* src2 = graph.data({dim_2, dim_4}, "src2");
    auto* dst = graph.data({dim_4}, "dst");

    REQUIRE_THROWS_AS(
        gt::sumprod_fiber(src1, src1, dst, axis_1, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::sumprod_fiber(src1, src2, src1, axis_1, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sumprod_fiber matches nntile::tensor::sumprod_fiber", "[graph][tensor]")
{
    const auto [src_shape, axis, redux, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, redux_none, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, redux_none, alpha_half, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, redux_none, alpha_one, beta_one});

    check_sumprod_fiber_vs_tensor_api<nntile::fp32_t>(
        src_shape, axis, redux, alpha, beta);
}
