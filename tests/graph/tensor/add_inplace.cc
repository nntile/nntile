/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/add_inplace.cc
 * Test TensorGraph add_inplace operation against nntile::tensor::add_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha = 1.0;
constexpr Scalar beta = 1.0;

} // anonymous namespace

template<typename T>
void check_add_inplace_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("add_inplace_test");
    auto* x_node = graph.data(shape, "x", DataType::FP32);
    auto* y_node = graph.data(shape, "y", DataType::FP32);
    x_node->mark_input(true);
    y_node->mark_input(true);
    y_node->mark_output(true);

    gt::add_inplace(alpha, x_node, beta, y_node);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("y");

    // --- Direct tensor API path (same input data) ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> src(traits, distr);
    nntile::tensor::Tensor<T> dst(traits, distr);

    {
        auto tile1 = src.get_tile(0);
        auto tile2 = dst.get_tile(0);
        auto loc1 = tile1.acquire(STARPU_W);
        auto loc2 = tile2.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc1[i] = static_cast<Y>(x_data[i]);
            loc2[i] = static_cast<Y>(y_data[i]);
        }
        loc1.release();
        loc2.release();
    }

    nntile::tensor::add_inplace<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph add_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* x = graph.data({dim0, dim1}, "x");
    auto* y = graph.data({dim0, dim1}, "y");

    gt::add_inplace(alpha, x, beta, y);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE("TensorGraph add_inplace rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* t = graph.data({4, 5}, "t");

    REQUIRE_THROWS_AS(gt::add_inplace(alpha, t, beta, t), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_inplace matches nntile::tensor::add_inplace", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 5}},
        std::tuple{0.5, -1.0, std::vector<Index>{6}},
        std::tuple{1.0, 2.0, std::vector<Index>{3, 4}});

    check_add_inplace_vs_tensor_api<nntile::fp32_t>(shape, alpha, beta);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 6}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 6}},
        std::tuple{0.5, -1.0, std::vector<Index>{6}},
        std::tuple{1.0, 2.0, std::vector<Index>{4, 4}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("add_inplace_untiled");
        auto* x_node = graph.data(shape, "x", DataType::FP32);
        auto* y_node = graph.data(shape, "y", DataType::FP32);
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::add_inplace(alpha, x_node, beta, y_node);

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
        TensorGraph graph("add_inplace_tiled");
        auto* x_node = graph.data(shape, "x", DataType::FP32);
        auto* y_node = graph.data(shape, "y", DataType::FP32);
        x_node->mark_input(true);
        y_node->mark_input(true);
        y_node->mark_output(true);

        gt::add_inplace(alpha, x_node, beta, y_node);
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
