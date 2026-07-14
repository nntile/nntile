/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/subtract_indexed_outputs.cc
 * Test TensorGraph subtract_indexed_outputs against
 * nntile::tensor::subtract_indexed_outputs.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/subtract_indexed_outputs.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/subtract_indexed_outputs.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cstdint>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar val = 1.0;
constexpr Index ignore_index = -1;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph subtract_indexed_outputs structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef labels = graph.data({4}, DataType::INT64);
    labels->set_name("labels");
    nntile::TensorRef dst = graph.data({4, 5});
    dst->set_name("dst");

    gt::subtract_indexed_outputs(val, labels, dst, ignore_index);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUBTRACT_INDEXED_OUTPUTS");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE(
    "TensorGraph subtract_indexed_outputs rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef labels = graph.data({4}, DataType::INT64);
    labels->set_name("labels");
    nntile::TensorRef dst = graph.data({4, 5});
    dst->set_name("dst");

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
    nntile::TensorRef labels = graph.data({4});
    labels->set_name("labels"); // FP32 default
    nntile::TensorRef dst = graph.data({5, 4});
    dst->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, dst, ignore_index),
        std::invalid_argument);
}

TEST_CASE("TensorGraph subtract_indexed_outputs rejects ndim mismatch",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef labels = graph.data({4}, DataType::INT64);
    labels->set_name("labels");
    // dst has ndim=3 (labels.ndim+2), but must be labels.ndim+1
    nntile::TensorRef dst = graph.data({5, 4, 3});
    dst->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, dst, ignore_index),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph subtract_indexed_outputs tiled matches untiled",
    "[graph][tensor]")
{
    const auto [labels_shape, n_class] =
        GENERATE(std::tuple{std::vector<Index>{4}, Index(6)},
            std::tuple{std::vector<Index>{2, 4}, Index(4)});

    std::vector<Index> dst_shape = labels_shape;
    dst_shape.push_back(n_class);
    const Index labels_nelems = std::accumulate(labels_shape.begin(),
        labels_shape.end(),
        Index(1),
        std::multiplies<>());
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<std::int64_t> labels_data(labels_nelems);
    std::vector<float> dst_data(dst_nelems);
    for (Index i = 0; i < labels_nelems; ++i)
    {
        labels_data[i] = static_cast<std::int64_t>(i % n_class);
    }
    for (Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = static_cast<float>(i);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("subtract_indexed_outputs_untiled");
        nntile::TensorRef labels_node = graph.data(labels_shape, DataType::INT64);
    labels_node->set_name("labels");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::subtract_indexed_outputs(val, labels_node, dst_node, ignore_index);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(labels_node, labels_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("subtract_indexed_outputs_tiled");
        nntile::TensorRef labels_node = graph.data(labels_shape, DataType::INT64);
    labels_node->set_name("labels");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::subtract_indexed_outputs(val, labels_node, dst_node, ignore_index);
        auto *nclass_axis = dst_node->axis(dst_node->ndim() - 1);
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

        runtime.bind_data(labels_node, labels_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
