/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/add_fiber.cc
 * Test TensorGraph add_fiber operation against nntile::tensor::add_fiber.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/add_fiber.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_fiber.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index batch_ndim_none = 0;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar beta_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Fiber shape for add_fiber: {tensor_shape[axis]} for batch_ndim=0
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
void check_add_fiber_vs_tensor_api(
    const std::vector<Index>& tensor_shape,
    Index axis,
    Index batch_ndim,
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
    TensorGraph graph("add_fiber_test");
    auto* fiber_node = graph.data(fiber_sh, "fiber", DataType::FP32);
    auto* tensor_node = graph.data(tensor_shape, "tensor", DataType::FP32);
    fiber_node->mark_input(true);
    tensor_node->mark_input(true);

    auto* out_node = gt::add_fiber(alpha, fiber_node, beta, tensor_node, "out",
                               axis, batch_ndim);
    out_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
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

    std::vector<float> graph_result = runtime.get_output<float>("out");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits fiber_traits(fiber_sh, fiber_sh);
    nntile::tensor::TensorTraits tensor_traits(tensor_shape, tensor_shape);
    std::vector<int> fiber_distr(fiber_traits.grid.nelems, distr_rank_single);
    std::vector<int> tensor_distr(tensor_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> fiber_t(fiber_traits, fiber_distr);
    nntile::tensor::Tensor<T> tensor_t(tensor_traits, tensor_distr);
    nntile::tensor::Tensor<T> out_t(tensor_traits, tensor_distr);

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

    nntile::tensor::add_fiber<T>(alpha, fiber_t, beta, tensor_t, out_t, axis, batch_ndim);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(tensor_nelems);
    {
        auto tile = out_t.get_tile(0);
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

TEST_CASE("TensorGraph add_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* fiber = graph.data({dim_4}, "fiber");
    auto* tensor = graph.data({dim_2, dim_4}, "tensor");

    auto* out = gt::add_fiber(alpha_one, fiber, beta_one, tensor, "out",
                         axis_1, batch_ndim_none);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD_FIBER");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE("TensorGraph add_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* fiber = graph.data({dim_4}, "fiber");
    auto* tensor = graph.data({dim_2, dim_4}, "tensor");

    REQUIRE_THROWS_AS(
        gt::add_fiber(alpha_one, fiber, beta_one, tensor, tensor, axis_1, batch_ndim_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_fiber matches nntile::tensor::add_fiber", "[graph][tensor]")
{
    const auto [tensor_shape, axis, batch_ndim, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, batch_ndim_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, batch_ndim_none, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, batch_ndim_none, alpha_one, beta_zero});

    check_add_fiber_vs_tensor_api<nntile::fp32_t>(
        tensor_shape, axis, batch_ndim, alpha, beta);
}
