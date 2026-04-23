/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sqrt.cc
 * Test TensorGraph sqrt operation against nntile::tensor::sqrt.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sqrt.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sqrt.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

template<typename T>
void check_sqrt_vs_tensor_api(
    const std::vector<Index>& shape)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sqrt_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* dst_node = gt::sqrt(src_node, "dst");
    dst_node->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    // Use positive values only (sqrt of negative gives NaN)
    std::vector<float> src_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> src(traits, distr);
    nntile::tensor::Tensor<T> dst(traits, distr);

    {
        auto tile = src.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    nntile::tensor::sqrt<T>(src, dst);
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

TEST_CASE("TensorGraph sqrt structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");

    auto* dst = gt::sqrt(src, "dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == dim0);
    REQUIRE(dst->shape()[1] == dim1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SQRT");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sqrt rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({4, 5}, "src");

    REQUIRE_THROWS_AS(gt::sqrt(src, src), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sqrt matches nntile::tensor::sqrt", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 5},
        std::vector<Index>{6},
        std::vector<Index>{2, 3},
        std::vector<Index>{1, 10});

    check_sqrt_vs_tensor_api<nntile::fp32_t>(shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sqrt tiled matches untiled", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    std::vector<float> untiled_result;
    {
        TensorGraph graph("sqrt_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        src_node->mark_input(true);
        auto* dst_node = gt::sqrt(src_node, "dst");
        dst_node->mark_output(true);
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("dst");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("sqrt_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        src_node->mark_input(true);
        auto* dst_node = gt::sqrt(src_node, "dst");
        dst_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>("dst");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
