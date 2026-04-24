/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/total_sum_accum.cc
 * Test TensorGraph total_sum_accum operation against nntile::tensor::total_sum_accum.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <cstdint>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/total_sum_accum.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/total_sum_accum.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr Index ignore_index = -1;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

// logsumexp and labels: same shape [batch]. src: [n_class, batch]. val: scalar.
template<typename T>
void check_total_sum_accum_vs_tensor_api(
    const std::vector<Index>& labels_shape,
    Index n_class)
{
    using Y = typename T::repr_t;
    std::vector<Index> src_shape = {n_class};
    src_shape.insert(src_shape.end(), labels_shape.begin(), labels_shape.end());

    const Index labels_nelems = std::accumulate(
        labels_shape.begin(), labels_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> logsumexp_data(labels_nelems);
    std::vector<float> src_data(src_nelems);
    std::vector<std::int64_t> labels_data(labels_nelems);
    std::vector<float> val_data(1, 0.0f);

    for(Index i = 0; i < labels_nelems; ++i)
    {
        logsumexp_data[i] = static_cast<float>(Y(i % 5));
        labels_data[i] = static_cast<std::int64_t>(i % n_class);
    }
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10));
    }

    // --- TensorGraph path ---
    TensorGraph graph("total_sum_accum_test");
    auto* logsumexp_node = graph.data(labels_shape, "logsumexp", DataType::FP32);
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* labels_node = graph.data(labels_shape, "labels", DataType::INT64);
    auto* val_node = graph.data({}, "val", DataType::FP32);
    logsumexp_node->mark_input(true);
    src_node->mark_input(true);
    labels_node->mark_input(true);
    val_node->mark_input(true);
    val_node->mark_output(true);

    gt::total_sum_accum(alpha_one, logsumexp_node, src_node, labels_node,
                   val_node, ignore_index);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    runtime.bind_data("logsumexp", logsumexp_data);
    runtime.bind_data("src", src_data);
    runtime.bind_data("labels", labels_data);
    runtime.bind_data("val", val_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("val");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits logsumexp_traits(labels_shape, labels_shape);
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits labels_traits(labels_shape, labels_shape);
    nntile::tensor::TensorTraits val_traits({}, {});
    std::vector<int> distr_single(1, distr_rank_single);
    std::vector<int> labels_distr(labels_traits.grid.nelems, distr_rank_single);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);

    nntile::tensor::Tensor<T> logsumexp_t(logsumexp_traits, labels_distr);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<nntile::int64_t> labels_t(labels_traits, labels_distr);
    nntile::tensor::Tensor<nntile::fp32_t> val_t(val_traits, distr_single);

    {
        auto tile = logsumexp_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < labels_nelems; ++i)
        {
            loc[i] = static_cast<Y>(logsumexp_data[i]);
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
    {
        auto tile = labels_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < labels_nelems; ++i)
        {
            loc[i] = labels_data[i];
        }
        loc.release();
    }
    {
        auto tile = val_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        loc[0] = val_data[0];
        loc.release();
    }

    nntile::tensor::total_sum_accum<T>(alpha_one, logsumexp_t, src_t, labels_t,
                              val_t, ignore_index);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(1);
    {
        auto tile = val_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        tensor_result[0] = static_cast<float>(loc[0]);
        loc.release();
    }

    REQUIRE(graph_result.size() == 1);
    REQUIRE(tensor_result.size() == 1);
    float diff = std::abs(graph_result[0] - tensor_result[0]);
    float ref = std::abs(tensor_result[0]) + 1e-10f;
    REQUIRE(diff / ref < tolerance);
}

TEST_CASE("TensorGraph total_sum_accum structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* logsumexp = graph.data({4}, "logsumexp");
    auto* src = graph.data({3, 4}, "src");
    auto* labels = graph.data({4}, "labels", DataType::INT64);
    auto* val = graph.data({}, "val", DataType::FP32);

    gt::total_sum_accum(alpha_one, logsumexp, src, labels, val, ignore_index);

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "TOTAL_SUM_ACCUM");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == val);
}

TEST_CASE("TensorGraph total_sum_accum rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* logsumexp = graph.data({4}, "logsumexp");
    auto* src = graph.data({3, 4}, "src");
    auto* labels = graph.data({4}, "labels", DataType::INT64);
    auto* val = graph.data({}, "val", DataType::FP32);

    REQUIRE_THROWS_AS(
        gt::total_sum_accum(alpha_one, nullptr, src, labels, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(alpha_one, logsumexp, nullptr, labels, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(alpha_one, logsumexp, src, nullptr, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(alpha_one, logsumexp, src, labels, nullptr, ignore_index),
        std::invalid_argument);
}

TEST_CASE("TensorGraph total_sum_accum rejects wrong dtypes", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* logsumexp = graph.data({4}, "logsumexp");
    auto* src = graph.data({3, 4}, "src");
    auto* labels = graph.data({4}, "labels");  // FP32 default
    auto* val = graph.data({}, "val", DataType::FP32);

    REQUIRE_THROWS_AS(
        gt::total_sum_accum(alpha_one, logsumexp, src, labels, val, ignore_index),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph total_sum_accum matches nntile::tensor::total_sum_accum",
    "[graph][tensor]")
{
    const auto [labels_shape, n_class] = GENERATE(
        std::tuple{std::vector<Index>{4}, Index(3)},
        std::tuple{std::vector<Index>{6}, Index(5)},
        std::tuple{std::vector<Index>{2, 3}, Index(4)});

    check_total_sum_accum_vs_tensor_api<nntile::fp32_t>(
        labels_shape, n_class);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph total_sum_accum tiled matches untiled", "[graph][tensor]")
{
    const auto [labels_shape, n_class] = GENERATE(
        std::tuple{std::vector<Index>{4}, Index(6)},
        std::tuple{std::vector<Index>{2, 4}, Index(4)});

    std::vector<Index> src_shape = {n_class};
    src_shape.insert(src_shape.end(), labels_shape.begin(), labels_shape.end());

    const Index labels_nelems = std::accumulate(
        labels_shape.begin(), labels_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> logsumexp_data(labels_nelems);
    std::vector<float> src_data(src_nelems);
    std::vector<std::int64_t> labels_data(labels_nelems);
    std::vector<float> val_data(1, 0.0f);

    for(Index i = 0; i < labels_nelems; ++i)
    {
        logsumexp_data[i] = static_cast<float>(i % 5);
        labels_data[i] = static_cast<std::int64_t>(i % n_class);
    }
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("total_sum_accum_untiled");
        auto* logsumexp_node = graph.data(labels_shape, "logsumexp", DataType::FP32);
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* labels_node = graph.data(labels_shape, "labels", DataType::INT64);
        auto* val_node = graph.data({}, "val", DataType::FP32);
        logsumexp_node->mark_input(true);
        src_node->mark_input(true);
        labels_node->mark_input(true);
        val_node->mark_input(true);
        val_node->mark_output(true);

        gt::total_sum_accum(alpha_one, logsumexp_node, src_node, labels_node,
                       val_node, ignore_index);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("logsumexp", logsumexp_data);
        runtime.bind_data("src", src_data);
        runtime.bind_data("labels", labels_data);
        runtime.bind_data("val", val_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("val");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("total_sum_accum_tiled");
        auto* logsumexp_node = graph.data(labels_shape, "logsumexp", DataType::FP32);
        auto* src_node = graph.data(src_shape, "src", DataType::FP32);
        auto* labels_node = graph.data(labels_shape, "labels", DataType::INT64);
        auto* val_node = graph.data({}, "val", DataType::FP32);
        logsumexp_node->mark_input(true);
        src_node->mark_input(true);
        labels_node->mark_input(true);
        val_node->mark_input(true);
        val_node->mark_output(true);

        gt::total_sum_accum(alpha_one, logsumexp_node, src_node, labels_node,
                       val_node, ignore_index);
        auto* nclass_axis = src_node->axis(0);
        for(auto* ag : graph.axis_groups())
        {
            if(ag == nclass_axis)
            {
                ag->set_tiling(ag->extent);
            }
            else
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("logsumexp", logsumexp_data);
        runtime.bind_data("src", src_data);
        runtime.bind_data("labels", labels_data);
        runtime.bind_data("val", val_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("val");
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
