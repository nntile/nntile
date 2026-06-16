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

} 

static std::vector<Index> make_src_shape(const std::vector<Index> &sin_shape)
{
    std::vector<Index> src_shape = sin_shape;
    src_shape.back() = sin_shape.back() * 2;
    return src_shape;
}

TEST_CASE("TensorGraph rope_backward structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *sin = graph.data({4, 2})->set_name("sin");
    auto *cos = graph.data({4, 2})->set_name("cos");
    auto *dy = graph.data({4, 4})->set_name("dy");
    auto *dx = gt::rope_backward(sin, cos, dy)->set_name("dx");

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
    auto *sin = graph.data({4, 2})->set_name("sin");
    auto *cos = graph.data({4, 2})->set_name("cos");
    auto *dy = graph.data({4, 4})->set_name("dy");

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
    const auto sin_shape =
        GENERATE(std::vector<Index>{4, 2}, std::vector<Index>{2, 3, 4});

    const std::vector<Index> dy_shape = make_src_shape(sin_shape);

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
        auto *sin_node =
            graph.data(sin_shape, DataType::FP32)->set_name("sin");
        auto *cos_node =
            graph.data(sin_shape, DataType::FP32)->set_name("cos");
        auto *dy_node = graph.data(dy_shape, DataType::FP32)->set_name("dy");
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        dy_node->mark_input(true);

        auto *dx_node =
            gt::rope_backward(sin_node, cos_node, dy_node)->set_name("dx");
        dx_node->mark_output(true);

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
        auto *sin_node =
            graph.data(sin_shape, DataType::FP32)->set_name("sin");
        auto *cos_node =
            graph.data(sin_shape, DataType::FP32)->set_name("cos");
        auto *dy_node = graph.data(dy_shape, DataType::FP32)->set_name("dy");
        sin_node->mark_input(true);
        cos_node->mark_input(true);
        dy_node->mark_input(true);

        auto *dx_node =
            gt::rope_backward(sin_node, cos_node, dy_node)->set_name("dx");
        dx_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

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