/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/subtract_indexed_outputs.cc
 * Test TensorGraph subtract_indexed_outputs against nntile::tensor::subtract_indexed_outputs.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <cstdint>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/subtract_indexed_outputs.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/subtract_indexed_outputs.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar val = 1.0;
constexpr Index ignore_index = -1;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

// dst shape: [n_class, ...labels_shape], labels.ndim = dst.ndim - 1
template<typename T>
void check_subtract_indexed_outputs_vs_tensor_api(
    const std::vector<Index>& labels_shape,
    Index n_class)
{
    using Y = typename T::repr_t;
    std::vector<Index> dst_shape = {n_class};
    dst_shape.insert(dst_shape.end(), labels_shape.begin(), labels_shape.end());
    const Index labels_nelems = std::accumulate(
        labels_shape.begin(), labels_shape.end(), Index(1), std::multiplies<>());
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<std::int64_t> labels_data(labels_nelems);
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < labels_nelems; ++i)
    {
        labels_data[i] = static_cast<std::int64_t>(i % n_class);
    }
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(i));
    }

    // --- TensorGraph path ---
    TensorGraph graph("subtract_indexed_outputs_test");
    auto* labels_node = graph.data(labels_shape, "labels", DataType::INT64);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    labels_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::subtract_indexed_outputs(val, labels_node, dst_node, ignore_index);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    runtime.bind_data("labels", labels_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits labels_traits(labels_shape, labels_shape);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> labels_distr(labels_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<nntile::int64_t> labels_t(labels_traits, labels_distr);
    nntile::tensor::Tensor<T> dst_t(dst_traits, dst_distr);

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
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::subtract_indexed_outputs<T>(val, labels_t, dst_t, ignore_index);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
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

TEST_CASE("TensorGraph subtract_indexed_outputs structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* labels = graph.data({4}, "labels", DataType::INT64);
    auto* dst = graph.data({5, 4}, "dst");

    gt::subtract_indexed_outputs(val, labels, dst, ignore_index);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUBTRACT_INDEXED_OUTPUTS");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph subtract_indexed_outputs rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* labels = graph.data({4}, "labels", DataType::INT64);
    auto* dst = graph.data({5, 4}, "dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, nullptr, dst, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, nullptr, ignore_index),
        std::invalid_argument);
}

TEST_CASE("TensorGraph subtract_indexed_outputs rejects non-INT64 labels",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* labels = graph.data({4}, "labels");  // FP32 default
    auto* dst = graph.data({5, 4}, "dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, dst, ignore_index),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph subtract_indexed_outputs matches nntile::tensor::subtract_indexed_outputs",
    "[graph][tensor]")
{
    const auto [labels_shape, n_class] = GENERATE(
        std::tuple{std::vector<Index>{4}, Index(5)},
        std::tuple{std::vector<Index>{6}, Index(3)},
        std::tuple{std::vector<Index>{2, 3}, Index(4)});

    check_subtract_indexed_outputs_vs_tensor_api<nntile::fp32_t>(
        labels_shape, n_class);
}
