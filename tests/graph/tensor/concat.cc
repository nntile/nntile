/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/concat.cc
 * Test TensorGraph concat operation and tile lowering.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>
#include <tuple>

#include "context_fixture.hh"
#include "nntile/graph/tensor/concat.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile/graph_runtime.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace layout = nntile::graph::tile_graph_layout_io;

namespace
{

Index shape_prod(const std::vector<Index>& shape)
{
    return std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());
}

//! Reference concat in Fortran flat layout (same as bind_data / get_output).
std::vector<float> reference_concat_fortran(const std::vector<Index>& a_shape,
    const std::vector<Index>& b_shape,
    Index axis,
    const std::vector<float>& a_data,
    const std::vector<float>& b_data)
{
    std::vector<Index> out_shape = a_shape;
    out_shape[static_cast<size_t>(axis)] =
        a_shape[static_cast<size_t>(axis)]
        + b_shape[static_cast<size_t>(axis)];
    const Index nelems = shape_prod(out_shape);
    std::vector<float> out(static_cast<size_t>(nelems));
    std::vector<Index> g;
    for(Index lin = 0; lin < nelems; ++lin)
    {
        layout::fortran_tile_linear_to_index(lin, out_shape, g);
        if(g[static_cast<size_t>(axis)]
            < a_shape[static_cast<size_t>(axis)])
        {
            out[static_cast<size_t>(lin)] =
                a_data[static_cast<size_t>(
                    layout::fortran_dense_linear_index(a_shape, g))];
        }
        else
        {
            std::vector<Index> gb = g;
            gb[static_cast<size_t>(axis)] -=
                a_shape[static_cast<size_t>(axis)];
            out[static_cast<size_t>(lin)] =
                b_data[static_cast<size_t>(
                    layout::fortran_dense_linear_index(b_shape, gb))];
        }
    }
    return out;
}

} // namespace

TEST_CASE("TensorGraph concat structure", "[graph][tensor]")
{
    constexpr Index d0 = 3;
    constexpr Index d1a = 2;
    constexpr Index d1b = 4;
    constexpr Index axis = 1;

    TensorGraph graph("concat_struct");
    auto* a = graph.data({d0, d1a}, "a", DataType::FP32);
    auto* b = graph.data({d0, d1b}, "b", DataType::FP32);
    auto* out = gt::concat(a, b, axis, "out");

    REQUIRE(out->shape()[0] == d0);
    REQUIRE(out->shape()[1] == d1a + d1b);
    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CONCAT");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->inputs()[0] == a);
    REQUIRE(ops[0]->inputs()[1] == b);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE("TensorGraph concat rejects invalid arguments", "[graph][tensor]")
{
    TensorGraph graph("concat_bad");
    auto* a = graph.data({2, 3}, "a", DataType::FP32);
    auto* b = graph.data({2, 3}, "b", DataType::FP32);

    REQUIRE_THROWS_AS(gt::concat(nullptr, b, 0, "o"), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::concat(a, nullptr, 0, "o"), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::concat(a, b, -1, "o"), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::concat(a, b, 2, "o"), std::invalid_argument);

    auto* b_bad = graph.data({3, 3}, "b2", DataType::FP32);
    REQUIRE_THROWS_AS(gt::concat(a, b_bad, 1, "o"), std::invalid_argument);

    TensorGraph other("other");
    auto* b_other = other.data({2, 3}, "bo", DataType::FP32);
    REQUIRE_THROWS_AS(gt::concat(a, b_other, 1, "o"), std::invalid_argument);

    auto* b_fp64 = graph.data({2, 3}, "bf64", DataType::FP64);
    REQUIRE_THROWS_AS(gt::concat(a, b_fp64, 1, "o"), std::invalid_argument);

    auto* b_1d = graph.data({6}, "b1d", DataType::FP32);
    REQUIRE_THROWS_AS(gt::concat(a, b_1d, 0, "o"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph concat matches Fortran reference (untiled)", "[graph][tensor]")
{
    using ShapesAxis =
        std::tuple<std::vector<Index>, std::vector<Index>, Index>;
    const ShapesAxis c = GENERATE(ShapesAxis{{5}, {3}, 0},
        ShapesAxis{{2, 4}, {2, 5}, 1},
        ShapesAxis{{3, 2}, {4, 2}, 0},
        ShapesAxis{{2, 2, 2}, {2, 2, 3}, 2});

    const auto& a_shape = std::get<0>(c);
    const auto& b_shape = std::get<1>(c);
    const Index axis = std::get<2>(c);

    TensorGraph graph("concat_ref");
    auto* a_node = graph.data(a_shape, "a", DataType::FP32);
    auto* b_node = graph.data(b_shape, "b", DataType::FP32);
    a_node->mark_input(true);
    b_node->mark_input(true);
    auto* out_node = gt::concat(a_node, b_node, axis, "out");
    out_node->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    const Index na = shape_prod(a_shape);
    const Index nb = shape_prod(b_shape);
    std::vector<float> a_data(static_cast<size_t>(na));
    std::vector<float> b_data(static_cast<size_t>(nb));
    for(Index i = 0; i < na; ++i)
    {
        a_data[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    for(Index i = 0; i < nb; ++i)
    {
        b_data[static_cast<size_t>(i)] = static_cast<float>(100 + i);
    }

    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> got = runtime.get_output<float>("out");
    std::vector<float> expect =
        reference_concat_fortran(a_shape, b_shape, axis, a_data, b_data);

    constexpr float tol = 1e-5f;
    REQUIRE(got.size() == expect.size());
    for(size_t i = 0; i < got.size(); ++i)
    {
        REQUIRE(std::abs(got[i] - expect[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph concat tiled matches untiled", "[graph][tensor]")
{
    const std::vector<Index> a_shape = {4, 6};
    const std::vector<Index> b_shape = {4, 5};
    constexpr Index axis = 1;

    std::vector<float> a_data(static_cast<size_t>(shape_prod(a_shape)));
    std::vector<float> b_data(static_cast<size_t>(shape_prod(b_shape)));
    for(size_t i = 0; i < a_data.size(); ++i)
    {
        a_data[i] = static_cast<float>(i * 3 + 7);
    }
    for(size_t i = 0; i < b_data.size(); ++i)
    {
        b_data[i] = static_cast<float>(50 - static_cast<int>(i));
    }

    std::vector<float> untiled_result;
    {
        TensorGraph graph("concat_untiled");
        auto* a_node = graph.data(a_shape, "a", DataType::FP32);
        auto* b_node = graph.data(b_shape, "b", DataType::FP32);
        a_node->mark_input(true);
        b_node->mark_input(true);
        auto* out_node = gt::concat(a_node, b_node, axis, "out");
        out_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("a", a_data);
        runtime.bind_data("b", b_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("out");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("concat_tiled");
        auto* a_node = graph.data(a_shape, "a", DataType::FP32);
        auto* b_node = graph.data(b_shape, "b", DataType::FP32);
        a_node->mark_input(true);
        b_node->mark_input(true);
        auto* out_node = gt::concat(a_node, b_node, axis, "out");
        out_node->mark_output(true);

        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("a", a_data);
        runtime.bind_data("b", b_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>("out");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
