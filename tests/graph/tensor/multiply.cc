/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/multiply.cc
 * Test TensorGraph multiply operation against nntile::tensor::multiply.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/multiply.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

template<typename T>
void check_multiply_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("multiply_test");
    auto* x_node = graph.data(shape, "x", DataType::FP32);
    auto* y_node = graph.data(shape, "y", DataType::FP32);
    x_node->mark_input(true);
    y_node->mark_input(true);

    auto* z_node = multiply(x_node, y_node, "z", alpha);
    z_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("z");

    // --- Direct tensor API path (same input data) ---
    tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    tensor::Tensor<T> src1(traits, distr);
    tensor::Tensor<T> src2(traits, distr);
    tensor::Tensor<T> dst(traits, distr);

    {
        auto tile1 = src1.get_tile(0);
        auto tile2 = src2.get_tile(0);
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

    tensor::multiply<T>(alpha, src1, src2, dst);
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

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph multiply structure", "[graph][tensor]")
{
    const auto alpha = GENERATE(1.0, 2.5, 0.5);
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* x = graph.data({dim0, dim1}, "x");
    auto* y = graph.data({dim0, dim1}, "y");

    auto* z = multiply(x, y, "z", alpha);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(z->shape()[0] == dim0);
    REQUIRE(z->shape()[1] == dim1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == z);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply matches tensor::multiply", "[graph][tensor]")
{
    const auto [alpha, shape] = GENERATE(
        std::tuple{1.0, std::vector<Index>{4, 5}},
        std::tuple{2.5, std::vector<Index>{4, 5}},
        std::tuple{0.5, std::vector<Index>{2, 3}},
        std::tuple{1.0, std::vector<Index>{6}},
        std::tuple{3.0, std::vector<Index>{1, 10}});

    check_multiply_vs_tensor_api<nntile::fp32_t>(
        shape, alpha);
}
