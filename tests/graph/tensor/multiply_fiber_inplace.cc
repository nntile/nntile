/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/multiply_fiber_inplace.cc
 * Test TensorGraph multiply_fiber_inplace operation against nntile::tensor::multiply_fiber_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/multiply_fiber_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_fiber_inplace.hh"
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

//! Fiber shape for multiply_fiber_inplace: {tensor_shape[axis]} (1D fiber)
static std::vector<Index> fiber_shape(
    const std::vector<Index>& tensor_shape,
    Index axis)
{
    return {tensor_shape[axis]};
}

template<typename T>
void check_multiply_fiber_inplace_vs_tensor_api(
    const std::vector<Index>& tensor_shape,
    Index axis,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index tensor_nelems = std::accumulate(
        tensor_shape.begin(), tensor_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> fiber_sh = fiber_shape(tensor_shape, axis);
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("multiply_fiber_inplace_test");
    auto* fiber_node = graph.data(fiber_sh, "fiber", DataType::FP32);
    auto* tensor_node = graph.data(tensor_shape, "tensor", DataType::FP32);
    fiber_node->mark_input(true);
    tensor_node->mark_input(true);
    tensor_node->mark_output(true);

    gt::multiply_fiber_inplace(alpha, fiber_node, tensor_node, axis);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> fiber_data(fiber_nelems);
    std::vector<float> tensor_data(tensor_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
    {
        fiber_data[i] = static_cast<float>(Y(i + 1));
    }
    for(Index i = 0; i < tensor_nelems; ++i)
    {
        tensor_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("fiber", fiber_data);
    runtime.bind_data("tensor", tensor_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("tensor");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits fiber_traits(fiber_sh, fiber_sh);
    nntile::tensor::TensorTraits tensor_traits(tensor_shape, tensor_shape);
    std::vector<int> fiber_distr(fiber_traits.grid.nelems, distr_rank_single);
    std::vector<int> tensor_distr(tensor_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> fiber_t(fiber_traits, fiber_distr);
    nntile::tensor::Tensor<T> tensor_t(tensor_traits, tensor_distr);

    {
        auto tile = fiber_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < fiber_nelems; ++i)
        {
            loc[i] = static_cast<Y>(fiber_data[i]);
        }
        loc.release();
    }
    {
        auto tile = tensor_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < tensor_nelems; ++i)
        {
            loc[i] = static_cast<Y>(tensor_data[i]);
        }
        loc.release();
    }

    nntile::tensor::multiply_fiber_inplace<T>(alpha, fiber_t, tensor_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(tensor_nelems);
    {
        auto tile = tensor_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < tensor_nelems; ++i)
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

TEST_CASE("TensorGraph multiply_fiber_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* fiber = graph.data({dim_4}, "fiber");
    auto* tensor = graph.data({dim_2, dim_4}, "tensor");

    gt::multiply_fiber_inplace(alpha_one, fiber, tensor, axis_1);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY_FIBER_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == tensor);
}

TEST_CASE("TensorGraph multiply_fiber_inplace rejects duplicate tensors",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* fiber = graph.data({dim_4}, "fiber");

    REQUIRE_THROWS_AS(
        gt::multiply_fiber_inplace(alpha_one, fiber, fiber, axis_0),
        std::invalid_argument);
}

TEST_CASE("TensorGraph multiply_fiber_inplace rejects mismatched shapes",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* fiber = graph.data({dim_4}, "fiber");
    auto* tensor = graph.data({dim_2, dim_4}, "tensor");

    // Fiber length must match tensor dim along axis
    auto* wrong_fiber = graph.data({dim_5}, "wrong_fiber");
    REQUIRE_THROWS_AS(
        gt::multiply_fiber_inplace(alpha_one, wrong_fiber, tensor, axis_1),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_fiber_inplace matches nntile::tensor::multiply_fiber_inplace",
    "[graph][tensor]")
{
    const auto [tensor_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, alpha_half},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, alpha_two});

    check_multiply_fiber_inplace_vs_tensor_api<nntile::fp32_t>(
        tensor_shape, axis, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_fiber_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [tensor_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Index(1), 1.0},
        std::tuple{std::vector<Index>{2, 4}, Index(0), 1.0});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> fiber_sh = fiber_shape(tensor_shape, axis);
    const Index tensor_nelems = std::accumulate(
        tensor_shape.begin(), tensor_shape.end(), Index(1), std::multiplies<>());
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> fiber_data(fiber_nelems);
    std::vector<float> tensor_data(tensor_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        fiber_data[i] = static_cast<float>(Y(i + 1));
    for(Index i = 0; i < tensor_nelems; ++i)
        tensor_data[i] = static_cast<float>(Y(-i - 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("multiply_fiber_inplace_untiled");
        auto* fiber_node = graph.data(fiber_sh, "fiber", DataType::FP32);
        auto* tensor_node = graph.data(tensor_shape, "tensor", DataType::FP32);
        fiber_node->mark_input(true);
        tensor_node->mark_input(true);
        tensor_node->mark_output(true);
        gt::multiply_fiber_inplace(alpha, fiber_node, tensor_node, axis);
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("fiber", fiber_data);
        runtime.bind_data("tensor", tensor_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("tensor");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("multiply_fiber_inplace_tiled");
        auto* fiber_node = graph.data(fiber_sh, "fiber", DataType::FP32);
        auto* tensor_node = graph.data(tensor_shape, "tensor", DataType::FP32);
        fiber_node->mark_input(true);
        tensor_node->mark_input(true);
        tensor_node->mark_output(true);
        gt::multiply_fiber_inplace(alpha, fiber_node, tensor_node, axis);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("fiber", fiber_data);
        runtime.bind_data("tensor", tensor_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>("tensor");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
