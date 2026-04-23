/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/rope.cc
 * Test TensorGraph rope operation against nntile::tensor::rope.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/rope.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/rope.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

// src.shape[0] = 2*sin.shape[0] (head_size)
template<typename T>
void check_rope_vs_tensor_api(const std::vector<Index>& sin_shape)
{
    using Y = typename T::repr_t;
    std::vector<Index> src_shape = {sin_shape[0] * 2};
    src_shape.insert(src_shape.end(), sin_shape.begin() + 1, sin_shape.end());

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(Y(i % 10) * 0.1);
        cos_data[i] = static_cast<float>(Y((i + 1) % 10) * 0.1);
    }
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10));
    }

    // --- TensorGraph path ---
    TensorGraph graph("rope_test");
    auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
    auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    sin_node->mark_input(true);
    cos_node->mark_input(true);
    src_node->mark_input(true);

    auto* dst_node = gt::rope(sin_node, cos_node, src_node, "dst");
    dst_node->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    runtime.bind_data("sin", sin_data);
    runtime.bind_data("cos", cos_data);
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits sin_traits(sin_shape, sin_shape);
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    std::vector<int> distr_single(1, distr_rank_single);
    std::vector<int> sin_distr(sin_traits.grid.nelems, distr_rank_single);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);

    nntile::tensor::Tensor<T> sin_t(sin_traits, sin_distr);
    nntile::tensor::Tensor<T> cos_t(sin_traits, sin_distr);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> dst_t(src_traits, src_distr);

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
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    nntile::tensor::rope<T>(sin_t, cos_t, src_t, dst_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(src_nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < src_nelems; ++i)
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

TEST_CASE("TensorGraph rope structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* sin = graph.data({2, 4}, "sin");
    auto* cos = graph.data({2, 4}, "cos");
    auto* src = graph.data({4, 4}, "src");
    auto* dst = gt::rope(sin, cos, src, "dst");

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == src->shape());

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ROPE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph rope rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* sin = graph.data({2, 4}, "sin");
    auto* cos = graph.data({2, 4}, "cos");
    auto* src = graph.data({4, 4}, "src");

    REQUIRE_THROWS_AS(gt::rope(nullptr, cos, src, "dst"), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::rope(sin, nullptr, src, "dst"), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::rope(sin, cos, nullptr, "dst"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope matches nntile::tensor::rope", "[graph][tensor]")
{
    const auto sin_shape = GENERATE(
        std::vector<Index>{2, 4},
        std::vector<Index>{4, 3, 2});

    check_rope_vs_tensor_api<nntile::fp32_t>(sin_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope tiled matches untiled", "[graph][tensor]")
{
    const auto sin_shape = GENERATE(
        std::vector<Index>{2, 4},
        std::vector<Index>{4, 3, 2});

    std::vector<Index> src_shape = {sin_shape[0] * 2};
    src_shape.insert(src_shape.end(), sin_shape.begin() + 1, sin_shape.end());

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(float(i % 10) * 0.1f);
        cos_data[i] = static_cast<float>(float((i + 1) % 10) * 0.1f);
    }
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("rope_untiled");
        auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
        auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        src_node->mark_input(true);

        auto* dst_node = gt::rope(sin_node, cos_node, src_node, "dst");
        dst_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("sin", sin_data);
        runtime.bind_data("cos", cos_data);
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("rope_tiled");
        auto* sin_node = graph.data(sin_shape, "sin", DataType::FP32);
        auto* cos_node = graph.data(sin_shape, "cos", DataType::FP32);
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        src_node->mark_input(true);

        auto* dst_node = gt::rope(sin_node, cos_node, src_node, "dst");
        dst_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("sin", sin_data);
        runtime.bind_data("cos", cos_data);
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("dst");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
