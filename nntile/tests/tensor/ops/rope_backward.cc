/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/rope_backward.cc
 * Test TensorGraph rope_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/rope_backward.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/rope_backward.hh"
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

//! RoPE pair axis is the last sin dim; dy uses 2x tiling on that axis.
void tile_rope_sin_cos_src(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *src)
{
    const Index rope_axis = sin->ndim() - 1;
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
        if (d == rope_axis)
        {
            std::vector<Index> src_seg;
            src_seg.reserve(sin_seg.size());
            for (Index v : sin_seg)
            {
                src_seg.push_back(2 * v);
            }
            src->axis(static_cast<int>(rope_axis))
                ->set_tiling(std::move(src_seg));
        }
        else
        {
            src->axis(static_cast<int>(d))->set_tiling(sin_seg);
        }
    }
}

} 

TEST_CASE("TensorGraph rope_backward structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef sin = graph.data({2});
    sin->set_name("sin");
    nntile::TensorRef cos = graph.data({2});
    cos->set_name("cos");
    nntile::TensorRef dy = graph.data({4});
    dy->set_name("dy");
    nntile::TensorRef dx = nntile::TensorRef::adopt(gt::rope_backward(sin, cos, dy));
    dx->set_name("dx");

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dx->shape() == dy->shape());

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ROPE_BACKWARD");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dx);
}

TEST_CASE("TensorGraph rope_backward rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef sin = graph.data({2});
    sin->set_name("sin");
    nntile::TensorRef cos = graph.data({2});
    cos->set_name("cos");
    nntile::TensorRef dy = graph.data({4});
    dy->set_name("dy");

    REQUIRE_THROWS_AS(
        gt::rope_backward(nullptr, cos, dy), std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::rope_backward(sin, nullptr, dy), std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::rope_backward(sin, cos, nullptr), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph rope_backward tiled matches untiled",
    "[graph][tensor]")
{
    const auto sin_shape = GENERATE(std::vector<Index>{2});

    std::vector<Index> dy_shape = {sin_shape[0] * 2};

    const Index sin_nelems = std::accumulate(
        sin_shape.begin(), sin_shape.end(), Index(1), std::multiplies<>());
    const Index dy_nelems = std::accumulate(
        dy_shape.begin(), dy_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> dy_data(dy_nelems);
    for (Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = static_cast<float>(float(i % 10) * 0.1f);
        cos_data[i] = static_cast<float>(float((i + 1) % 10) * 0.1f);
    }
    for (Index i = 0; i < dy_nelems; ++i)
    {
        dy_data[i] = static_cast<float>(i % 10);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("rope_backward_untiled");
        nntile::TensorRef sin_node = graph.data(sin_shape, DataType::FP32);
    sin_node->set_name("sin");
        nntile::TensorRef cos_node = graph.data(sin_shape, DataType::FP32);
    cos_node->set_name("cos");
        nntile::TensorRef dy_node = graph.data(dy_shape, DataType::FP32);
    dy_node->set_name("dy");

        nntile::TensorRef dx_node = nntile::TensorRef::adopt(gt::rope_backward(sin_node, cos_node, dy_node));
    dx_node->set_name("dx");

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(sin_node, sin_data);
        runtime.bind_data(cos_node, cos_data);
        runtime.bind_data(dy_node, dy_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dx_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("rope_backward_tiled");
        nntile::TensorRef sin_node = graph.data(sin_shape, DataType::FP32);
    sin_node->set_name("sin");
        nntile::TensorRef cos_node = graph.data(sin_shape, DataType::FP32);
    cos_node->set_name("cos");
        nntile::TensorRef dy_node = graph.data(dy_shape, DataType::FP32);
    dy_node->set_name("dy");

        nntile::TensorRef dx_node = nntile::TensorRef::adopt(gt::rope_backward(sin_node, cos_node, dy_node));
    dx_node->set_name("dx");
        tile_rope_sin_cos_src(sin_node, cos_node, dy_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(sin_node, sin_data);
        runtime.bind_data(cos_node, cos_data);
        runtime.bind_data(dy_node, dy_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(dx_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
