/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/scatter.cc
 * Test TensorGraph scatter operation against nntile::tensor::scatter.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/scatter.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_scatter_vs_tensor_api(const std::vector<Index>& shape)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("scatter_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::scatter(src_node, dst_node);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems, 0.0f);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i * 2 - 3));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (single-tile: scatter = copy) ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(traits, distr);
    nntile::tensor::Tensor<T> dst_t(traits, distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::scatter<T>(src_t, dst_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
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

TEST_CASE("TensorGraph scatter structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({4, 5}, "src");
    auto* dst = graph.data({4, 5}, "dst");
    gt::scatter(src, dst);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SCATTER");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph scatter rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({4, 5}, "src");
    auto* dst = graph.data({4, 5}, "dst");

    REQUIRE_THROWS_AS(gt::scatter(nullptr, dst), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::scatter(src, nullptr), std::invalid_argument);
}

TEST_CASE("TensorGraph scatter rejects shape mismatch", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({4, 5}, "src");
    auto* dst = graph.data({3, 4}, "dst");

    REQUIRE_THROWS_AS(gt::scatter(src, dst), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scatter matches nntile::tensor::scatter", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 5},
        std::vector<Index>{6},
        std::vector<Index>{2, 3});

    check_scatter_vs_tensor_api<nntile::fp32_t>(shape);
}

// scatter requires src to be single-tiled; tiling all shared axes would
// violate that constraint, so no tiled-vs-untiled test is added here.
