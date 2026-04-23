/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/rope_backward.cc
 * Test TensorGraph rope_backward operation against nntile::tensor::rope_backward.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/rope_backward.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/rope_backward.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_rope_backward_vs_tensor_api(const std::vector<Index>& sin_shape)
{
    using Y = typename T::repr_t;
    std::vector<Index> dy_shape = {sin_shape[0] * 2};
    dy_shape.insert(dy_shape.end(), sin_shape.begin() + 1, sin_shape.end());

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index dy_nelems = std::accumulate(
        dy_shape.begin(), dy_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> dy_data(dy_nelems);
    for(Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(Y(i % 10) * 0.1);
        cos_data[i] = static_cast<float>(Y((i + 1) % 10) * 0.1);
    }
    for(Index i = 0; i < dy_nelems; ++i)
    {
        dy_data[i] = static_cast<float>(Y(i % 10));
    }

    // --- TensorGraph path ---
    TensorGraph graph("rope_backward_test");
    auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
    auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
    auto* dy_node = graph.data(dy_shape, "dy", DataType::FP32);
    sin_node->mark_input(true);
    cos_node->mark_input(true);
    dy_node->mark_input(true);

    auto* dx_node = gt::rope_backward(sin_node, cos_node, dy_node, "dx");
    dx_node->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    runtime.bind_data("sin", sin_data);
    runtime.bind_data("cos", cos_data);
    runtime.bind_data("dy", dy_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dx");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits sin_traits(sin_shape, sin_shape);
    nntile::tensor::TensorTraits dy_traits(dy_shape, dy_shape);
    std::vector<int> sin_distr(sin_traits.grid.nelems, distr_rank_single);
    std::vector<int> dy_distr(dy_traits.grid.nelems, distr_rank_single);

    nntile::tensor::Tensor<T> sin_t(sin_traits, sin_distr);
    nntile::tensor::Tensor<T> cos_t(sin_traits, sin_distr);
    nntile::tensor::Tensor<T> dy_t(dy_traits, dy_distr);
    nntile::tensor::Tensor<T> dx_t(dy_traits, dy_distr);

    {
        auto tile = sin_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < sin_nelems; ++i)
        {
            loc[i] = static_cast<Y>(sin_data[i]);
        }
        loc.release();
    }
    {
        auto tile = cos_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < sin_nelems; ++i)
        {
            loc[i] = static_cast<Y>(cos_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dy_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < dy_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dy_data[i]);
        }
        loc.release();
    }

    nntile::tensor::rope_backward<T>(sin_t, cos_t, dy_t, dx_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dy_nelems);
    {
        auto tile = dx_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dy_nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        float diff = std::abs(graph_result[i] - tensor_result[i]);
        float ref = std::abs(tensor_result[i]) + 1e-10f;
        REQUIRE(diff / ref < tolerance);
    }
}

TEST_CASE("TensorGraph rope_backward structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* sin = graph.data({2, 4}, "sin");
    auto* cos = graph.data({2, 4}, "cos");
    auto* dy = graph.data({4, 4}, "dy");
    auto* dx = gt::rope_backward(sin, cos, dy, "dx");

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dx->shape() == dy->shape());

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ROPE_BACKWARD");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dx);
}

TEST_CASE("TensorGraph rope_backward rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* sin = graph.data({2, 4}, "sin");
    auto* cos = graph.data({2, 4}, "cos");
    auto* dy = graph.data({4, 4}, "dy");

    REQUIRE_THROWS_AS(
        gt::rope_backward(nullptr, cos, dy, "dx"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::rope_backward(sin, nullptr, dy, "dx"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::rope_backward(sin, cos, nullptr, "dx"),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope_backward matches nntile::tensor::rope_backward", "[graph][tensor]")
{
    const auto sin_shape = GENERATE(
        std::vector<Index>{2, 4},
        std::vector<Index>{4, 3, 2});

    check_rope_backward_vs_tensor_api<nntile::fp32_t>(sin_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope_backward tiled matches untiled", "[graph][tensor]")
{
    const auto sin_shape = GENERATE(
        std::vector<Index>{2, 4},
        std::vector<Index>{4, 3, 2});

    std::vector<Index> dy_shape = {sin_shape[0] * 2};
    dy_shape.insert(dy_shape.end(), sin_shape.begin() + 1, sin_shape.end());

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index dy_nelems = std::accumulate(
        dy_shape.begin(), dy_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> dy_data(dy_nelems);
    for(Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(float(i % 10) * 0.1f);
        cos_data[i] = static_cast<float>(float((i + 1) % 10) * 0.1f);
    }
    for(Index i = 0; i < dy_nelems; ++i)
    {
        dy_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("rope_backward_untiled");
        auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
        auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
        auto* dy_node = graph.data(dy_shape, "dy", DataType::FP32);
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        dy_node->mark_input(true);

        auto* dx_node = gt::rope_backward(sin_node, cos_node, dy_node, "dx");
        dx_node->mark_output(true);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("sin", sin_data);
        runtime.bind_data("cos", cos_data);
        runtime.bind_data("dy", dy_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dx");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("rope_backward_tiled");
        auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
        auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
        auto* dy_node = graph.data(dy_shape, "dy", DataType::FP32);
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        dy_node->mark_input(true);

        auto* dx_node = gt::rope_backward(sin_node, cos_node, dy_node, "dx");
        dx_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("sin", sin_data);
        runtime.bind_data("cos", cos_data);
        runtime.bind_data("dy", dy_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("dx");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
