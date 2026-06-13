/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/rope.cc
 * Test TensorGraph rope operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/rope.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/rope.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

//! RoPE head axis is 2x sin/cos; keep src dim0 tiles at 2x sin dim0 tiles.
void tile_rope_sin_cos_src(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *src)
{
    for (Index d = 0; d < sin->ndim(); ++d)
    {
        const Index extent = sin->shape()[static_cast<size_t>(d)];
        const Index tile = (extent + 1) / 2;
        std::vector<Index> sin_seg;
        if (extent <= tile)
        {
            sin_seg = {extent};
        }
        else
        {
            sin_seg = {tile, extent - tile};
        }
        sin->axis(static_cast<int>(d))->set_tiling(sin_seg);
        cos->axis(static_cast<int>(d))->set_tiling(sin_seg);
        if (d == 0)
        {
            std::vector<Index> src_seg;
            src_seg.reserve(sin_seg.size());
            for (Index v : sin_seg)
            {
                src_seg.push_back(2 * v);
            }
            src->axis(0)->set_tiling(std::move(src_seg));
        }
        else
        {
            src->axis(static_cast<int>(d))->set_tiling(sin_seg);
        }
    }
}

} 

TEST_CASE("TensorGraph rope structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *sin = graph.data({2, 4})->set_name("sin");
    auto *cos = graph.data({2, 4})->set_name("cos");
    auto *src = graph.data({4, 4})->set_name("src");
    auto *dst = gt::rope(sin, cos, src);

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == src->shape());

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ROPE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph rope rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *sin = graph.data({2, 4})->set_name("sin");
    auto *cos = graph.data({2, 4})->set_name("cos");
    auto *src = graph.data({4, 4})->set_name("src");

    REQUIRE_THROWS_AS(gt::rope(nullptr, cos, src), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::rope(sin, nullptr, src), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::rope(sin, cos, nullptr), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope tiled matches untiled",
    "[graph][tensor]")
{
    const auto sin_shape =
        GENERATE(std::vector<Index>{2, 4}, std::vector<Index>{4, 3, 2});

    std::vector<Index> src_shape = {sin_shape[0] * 2};
    src_shape.insert(src_shape.end(), sin_shape.begin() + 1, sin_shape.end());

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> src_data(src_nelems);
    for (Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(float(i % 10) * 0.1f);
        cos_data[i] = static_cast<float>(float((i + 1) % 10) * 0.1f);
    }
    for (Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("rope_untiled");
        auto *sin_node =
            graph.data(sin_shape, DataType::FP32)->set_name("sin");
        auto *cos_node =
            graph.data(sin_shape, DataType::FP32)->set_name("cos");
        auto *src_node =
            graph.data(src_shape, DataType::FP32)->set_name("src");
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        src_node->mark_input(true);

        auto *dst_node = gt::rope(sin_node, cos_node, src_node);
        dst_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(sin_node, sin_data);
        runtime.bind_data(cos_node, cos_data);
        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("rope_tiled");
        auto *sin_node =
            graph.data(sin_shape, DataType::FP32)->set_name("sin");
        auto *cos_node =
            graph.data(sin_shape, DataType::FP32)->set_name("cos");
        auto *src_node =
            graph.data(src_shape, DataType::FP32)->set_name("src");
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        src_node->mark_input(true);

        auto *dst_node = gt::rope(sin_node, cos_node, src_node);
        dst_node->mark_output(true);
        tile_rope_sin_cos_src(sin_node, cos_node, src_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(sin_node, sin_data);
        runtime.bind_data(cos_node, cos_data);
        runtime.bind_data(src_node, src_data);
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
