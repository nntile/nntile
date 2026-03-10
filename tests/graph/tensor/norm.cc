/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/norm.cc
 * Test TensorGraph norm operation against nntile::tensor::norm.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/norm.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm.hh"
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
void check_norm_vs_tensor_api(
    const std::vector<Index>& x_shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("norm_test");
    auto* x_node = graph.data(x_shape, "x", DataType::FP32);
    auto* y_node = graph.data({}, "y", DataType::FP32);  // scalar
    x_node->mark_input(true);
    y_node->mark_input(true);
    y_node->mark_output(true);

    gt::norm(x_node, y_node, alpha, beta);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(x_nelems);
    std::vector<float> y_data(1);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
    }
    y_data[0] = (beta != beta_zero) ? 1.0f : 0.0f;

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("y");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits x_traits(x_shape, x_shape);
    nntile::tensor::TensorTraits y_traits({}, {});
    std::vector<int> x_distr(x_traits.grid.nelems, distr_rank_single);
    std::vector<int> y_distr(1, distr_rank_single);
    nntile::tensor::Tensor<T> x_t(x_traits, x_distr);
    nntile::tensor::Tensor<T> y_t(y_traits, y_distr);

    {
        auto tile = x_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < x_nelems; ++i)
        {
            loc[i] = static_cast<Y>(x_data[i]);
        }
        loc.release();
    }
    {
        auto tile = y_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        loc[0] = static_cast<Y>(y_data[0]);
        loc.release();
    }

    nntile::tensor::norm<T>(alpha, x_t, beta, y_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(1);
    {
        auto tile = y_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        tensor_result[0] = static_cast<float>(loc[0]);
        loc.release();
    }

    REQUIRE(graph_result.size() == 1);
    REQUIRE(tensor_result.size() == 1);
    REQUIRE(std::abs(graph_result[0] - tensor_result[0]) < tolerance);
}

TEST_CASE("TensorGraph norm structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* x = graph.data({dim0, dim1}, "x");
    auto* y = graph.data({}, "y");

    gt::norm(x, y, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(y->shape().empty());

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "NORM");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE("TensorGraph norm rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* t = graph.data({4, 5}, "t");

    REQUIRE_THROWS_AS(gt::norm(t, t, alpha_one, beta_zero), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm matches nntile::tensor::norm", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 0.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 0.0, std::vector<Index>{6}},
        std::tuple{1.0, 1.0, std::vector<Index>{3, 4}});

    check_norm_vs_tensor_api<nntile::fp32_t>(shape, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm tiled matches untiled", "[graph][tensor]")
{
    const auto [alpha, beta, x_shape] = GENERATE(
        std::tuple{1.0, 0.0, std::vector<Index>{4, 6}},
        std::tuple{1.0, 1.0, std::vector<Index>{6}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
    }
    std::vector<float> y_data(1);
    y_data[0] = (beta != beta_zero) ? 1.0f : 0.0f;

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("norm_untiled");
        auto* x_node = graph.data(x_shape, "x", DataType::FP32);
        auto* y_node = graph.data({}, "y", DataType::FP32);
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::norm(x_node, y_node, alpha, beta);

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("x", x_data);
        runtime.bind_data("y", y_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("y");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("norm_tiled");
        auto* x_node = graph.data(x_shape, "x", DataType::FP32);
        auto* y_node = graph.data({}, "y", DataType::FP32);
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::norm(x_node, y_node, alpha, beta);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("x", x_data);
        runtime.bind_data("y", y_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("y");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
