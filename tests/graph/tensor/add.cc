/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/add.cc
 * Test TensorGraph add operation against nntile::tensor::add.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <numeric>

#include "nntile/context.hh"
#include "nntile/graph/tensor/add.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add.hh"
#include "nntile/tensor/fill.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

//! Run add via TensorGraph (compile + execute) and via tensor API, compare
template<typename T>
void check_add_vs_tensor_api(const std::vector<Index>& shape,
                             Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    const Index nelems =
        std::accumulate(shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("add_test");
    auto* x_node = graph.data(shape, "x", DataType::FP32);
    auto* y_node = graph.data(shape, "y", DataType::FP32);
    x_node->mark_input(true);
    y_node->mark_input(true);

    auto* z_node = add(alpha, x_node, beta, y_node, "z");
    z_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    // Generate input data once
    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    // --- TensorGraph path ---
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

    tensor::add<T>(alpha, src1, beta, src2, dst);
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
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < 1e-5f);
    }
}

TEST_CASE("TensorGraph add structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");

    auto* z = add(1.0, x, 1.0, y, "z");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(z->shape()[0] == 4);
    REQUIRE(z->shape()[1] == 5);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == z);
}

TEST_CASE("TensorGraph add matches tensor::add", "[graph][tensor]")
{
    Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0);

    SECTION("alpha=1, beta=1, shape {4,5}")
    {
        check_add_vs_tensor_api<nntile::fp32_t>({4, 5}, 1.0, 1.0);
    }
    SECTION("alpha=2, beta=3, shape {4,5}")
    {
        check_add_vs_tensor_api<nntile::fp32_t>({4, 5}, 2.0, 3.0);
    }
    SECTION("alpha=0.5, beta=-1, shape {6}")
    {
        check_add_vs_tensor_api<nntile::fp32_t>({6}, 0.5, -1.0);
    }
}
