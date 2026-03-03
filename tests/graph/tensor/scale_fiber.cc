/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/scale_fiber.cc
 * Test TensorGraph scale_fiber operation against nntile::tensor::scale_fiber.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/scale_fiber.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_fiber.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index batch_ndim_none = 0;
constexpr Scalar alpha = 2.5;
constexpr Scalar alpha_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Fiber shape: {dst_shape[axis]} for batch_ndim=0
static std::vector<Index> fiber_shape(
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim)
{
    std::vector<Index> out;
    out.reserve(batch_ndim + 1);
    out.push_back(dst_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(dst_shape[dst_shape.size() - batch_ndim + i]);
    }
    return out;
}

template<typename T>
void check_scale_fiber_vs_tensor_api(
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim,
    Scalar alpha_val)
{
    using Y = typename T::repr_t;
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> fiber_sh = fiber_shape(dst_shape, axis, batch_ndim);
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("scale_fiber_test");
    auto* src_node = graph.data(fiber_sh, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* dst_node = scale_fiber(alpha_val, src_node, "dst", dst_shape,
                                 axis, batch_ndim);
    dst_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(fiber_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    tensor::TensorTraits src_traits(fiber_sh, fiber_sh);
    tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    tensor::Tensor<T> src_t(src_traits, src_distr);
    tensor::Tensor<T> dst_t(dst_traits, dst_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < fiber_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    tensor::scale_fiber<T>(alpha_val, src_t, dst_t, axis, batch_ndim);
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

TEST_CASE("TensorGraph scale_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_4}, "src");

    auto* dst = scale_fiber(alpha, src, "dst", {dim_2, dim_4},
                           axis_1, batch_ndim_none);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SCALE_FIBER");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph scale_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_4}, "src");

    REQUIRE_THROWS_AS(
        scale_fiber(alpha, src, src, axis_1, batch_ndim_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scale_fiber matches tensor::scale_fiber", "[graph][tensor]")
{
    const auto [dst_shape, axis, batch_ndim, alpha_val] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, batch_ndim_none, alpha},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, batch_ndim_none, alpha},
        std::tuple{std::vector<Index>{dim_4, dim_5}, axis_1, batch_ndim_none, alpha_one});

    check_scale_fiber_vs_tensor_api<nntile::fp32_t>(
        dst_shape, axis, batch_ndim, alpha_val);
}
