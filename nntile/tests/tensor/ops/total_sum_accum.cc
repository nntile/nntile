/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/total_sum_accum.cc
 * Test TensorGraph total_sum_accum operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/total_sum_accum.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/total_sum_accum.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cstdint>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr Index ignore_index = -1;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph total_sum_accum structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef logsumexp = graph.data({4});
    logsumexp->set_name("logsumexp");
    nntile::TensorRef src = graph.data({4, 3});
    src->set_name("src");
    nntile::TensorRef labels = graph.data({4}, DataType::INT64);
    labels->set_name("labels");
    nntile::TensorRef val = graph.data({}, DataType::FP32);
    val->set_name("val");

    gt::total_sum_accum(alpha_one, logsumexp, src, labels, val, ignore_index);

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "TOTAL_SUM_ACCUM");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == val);
}

TEST_CASE("TensorGraph total_sum_accum rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef logsumexp = graph.data({4});
    logsumexp->set_name("logsumexp");
    nntile::TensorRef src = graph.data({4, 3});
    src->set_name("src");
    nntile::TensorRef labels = graph.data({4}, DataType::INT64);
    labels->set_name("labels");
    nntile::TensorRef val = graph.data({}, DataType::FP32);
    val->set_name("val");

    REQUIRE_THROWS_AS(gt::total_sum_accum(
                          alpha_one, nullptr, src, labels, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(
            alpha_one, logsumexp, nullptr, labels, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(
            alpha_one, logsumexp, src, nullptr, val, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::total_sum_accum(
            alpha_one, logsumexp, src, labels, nullptr, ignore_index),
        std::invalid_argument);
}

TEST_CASE(
    "TensorGraph total_sum_accum rejects wrong dtypes", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef logsumexp = graph.data({4});
    logsumexp->set_name("logsumexp");
    nntile::TensorRef src = graph.data({4, 3});
    src->set_name("src");
    nntile::TensorRef labels = graph.data({4});
    labels->set_name("labels"); // FP32 default
    nntile::TensorRef val = graph.data({}, DataType::FP32);
    val->set_name("val");

    REQUIRE_THROWS_AS(
        gt::total_sum_accum(
            alpha_one, logsumexp, src, labels, val, ignore_index),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph total_sum_accum tiled matches untiled",
    "[graph][tensor]")
{
    const auto [labels_shape, n_class] =
        GENERATE(std::tuple{std::vector<Index>{4}, Index(6)},
            std::tuple{std::vector<Index>{2, 4}, Index(4)});

    std::vector<Index> src_shape = labels_shape;
    src_shape.push_back(n_class);

    const Index labels_nelems = std::accumulate(labels_shape.begin(),
        labels_shape.end(),
        Index(1),
        std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> logsumexp_data(labels_nelems);
    std::vector<float> src_data(src_nelems);
    std::vector<std::int64_t> labels_data(labels_nelems);
    std::vector<float> val_data(1, 0.0f);

    for (Index i = 0; i < labels_nelems; ++i)
    {
        logsumexp_data[i] = static_cast<float>(i % 5);
        labels_data[i] = static_cast<std::int64_t>(i % n_class);
    }
    for (Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("total_sum_accum_untiled");
        nntile::TensorRef logsumexp_node = graph.data(labels_shape, DataType::FP32);
    logsumexp_node->set_name("logsumexp");
        nntile::TensorRef src_node = graph.data(src_shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef labels_node = graph.data(labels_shape, DataType::INT64);
    labels_node->set_name("labels");
        nntile::TensorRef val_node = graph.data({}, DataType::FP32);
    val_node->set_name("val");

        gt::total_sum_accum(alpha_one,
            logsumexp_node,
            src_node,
            labels_node,
            val_node,
            ignore_index);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(logsumexp_node, logsumexp_data);
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(labels_node, labels_data);
        runtime.bind_data(val_node, val_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(val_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("total_sum_accum_tiled");
        nntile::TensorRef logsumexp_node = graph.data(labels_shape, DataType::FP32);
    logsumexp_node->set_name("logsumexp");
        nntile::TensorRef src_node = graph.data(src_shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef labels_node = graph.data(labels_shape, DataType::INT64);
    labels_node->set_name("labels");
        nntile::TensorRef val_node = graph.data({}, DataType::FP32);
    val_node->set_name("val");

        gt::total_sum_accum(alpha_one,
            logsumexp_node,
            src_node,
            labels_node,
            val_node,
            ignore_index);
        auto *nclass_axis = src_node->axis(src_node->ndim() - 1);
        for (auto *ag : graph.axis_groups())
        {
            if (ag == nclass_axis)
            {
                ag->set_tiling(ag->extent);
            }
            else
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(logsumexp_node, logsumexp_data);
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(labels_node, labels_data);
        runtime.bind_data(val_node, val_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(val_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
