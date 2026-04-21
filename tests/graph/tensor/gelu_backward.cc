/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/gelu_backward.cc
 * Test TensorGraph gelu_backward operation against nntile::tensor::gelu_backward.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/gelu_backward.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu_backward.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

template<typename T>
void check_gelu_backward_vs_tensor_api(
    const std::vector<Index>& shape)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("gelu_backward_test");
    auto* x_node = graph.data(shape, "x", DataType::FP32);
    auto* dy_node = graph.data(shape, "dy", DataType::FP32);
    auto* dx_node = graph.data(shape, "dx", DataType::FP32);
    x_node->mark_input(true);
    dy_node->mark_input(true);
    dx_node->mark_input(true);
    dx_node->mark_output(true);

    gt::gelu_backward(x_node, dy_node, dx_node);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(nelems), dy_data(nelems), dx_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i - nelems / 2));
        dy_data[i] = static_cast<float>(Y(i % 7 + 1));
        dx_data[i] = 0.0f;
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("dy", dy_data);
    runtime.bind_data("dx", dx_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dx");

    // --- Direct tensor API path (same input data) ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> x_t(traits, distr);
    nntile::tensor::Tensor<T> dy_t(traits, distr);
    nntile::tensor::Tensor<T> dx_t(traits, distr);

    {
        auto tile = x_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
            loc[i] = static_cast<Y>(x_data[i]);
        loc.release();
    }
    {
        auto tile = dy_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
            loc[i] = static_cast<Y>(dy_data[i]);
        loc.release();
    }
    {
        auto tile = dx_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
            loc[i] = static_cast<Y>(dx_data[i]);
        loc.release();
    }

    nntile::tensor::gelu_backward<T>(x_t, dy_t, dx_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dx_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
            tensor_result[i] = static_cast<float>(loc[i]);
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph gelu_backward structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* x = graph.data({dim0, dim1}, "x");
    auto* dy = graph.data({dim0, dim1}, "dy");
    auto* dx = graph.data({dim0, dim1}, "dx");

    gt::gelu_backward(x, dy, dx);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "GELU_BACKWARD");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dx);
}

TEST_CASE("TensorGraph gelu_backward rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* x = graph.data({4, 5}, "x");
    auto* dy = graph.data({4, 5}, "dy");

    REQUIRE_THROWS_AS(gt::gelu_backward(x, x, dy), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::gelu_backward(x, dy, x), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::gelu_backward(x, dy, dy), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gelu_backward matches nntile::tensor::gelu_backward", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 5},
        std::vector<Index>{6},
        std::vector<Index>{2, 3},
        std::vector<Index>{1, 10});

    check_gelu_backward_vs_tensor_api<nntile::fp32_t>(shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gelu_backward tiled matches untiled", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(nelems), dy_data(nelems), dx_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i - nelems / 2));
        dy_data[i] = static_cast<float>(Y(i % 7 + 1));
        dx_data[i] = 0.0f;
    }

    std::vector<float> untiled_result;
    {
        TensorGraph graph("gelu_backward_untiled");
        auto* x_node = graph.data(shape, "x", DataType::FP32);
        auto* dy_node = graph.data(shape, "dy", DataType::FP32);
        auto* dx_node = graph.data(shape, "dx", DataType::FP32);
        x_node->mark_input(true);
        dy_node->mark_input(true);
        dx_node->mark_input(true);
        dx_node->mark_output(true);
        gt::gelu_backward(x_node, dy_node, dx_node);

        TensorGraph::Runtime runtime(graph);
        runtime.compile();
        runtime.bind_data("x", x_data);
        runtime.bind_data("dy", dy_data);
        runtime.bind_data("dx", dx_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("dx");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("gelu_backward_tiled");
        auto* x_node = graph.data(shape, "x", DataType::FP32);
        auto* dy_node = graph.data(shape, "dy", DataType::FP32);
        auto* dx_node = graph.data(shape, "dx", DataType::FP32);
        x_node->mark_input(true);
        dy_node->mark_input(true);
        dx_node->mark_input(true);
        dx_node->mark_output(true);
        gt::gelu_backward(x_node, dy_node, dx_node);

        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TensorGraph::Runtime runtime(graph);
        runtime.compile();
        runtime.bind_data("x", x_data);
        runtime.bind_data("dy", dy_data);
        runtime.bind_data("dx", dx_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>("dx");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
