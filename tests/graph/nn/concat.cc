/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/concat.cc
 * Test NNGraph concat (forward + tile lowering; backward not supported).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>
#include <tuple>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph.hh"
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph concat structure", "[graph][nn_graph]")
{
    constexpr Index d0 = 3;
    constexpr Index d1a = 2;
    constexpr Index d1b = 4;
    constexpr Index axis = 1;

    NNGraph g("nn_concat_struct");
    auto* a = g.tensor({d0, d1a}, "a", DataType::FP32);
    auto* b = g.tensor({d0, d1b}, "b", DataType::FP32);
    auto* out = concat(a, b, axis, "out");

    REQUIRE(out->shape()[0] == d0);
    REQUIRE(out->shape()[1] == d1a + d1b);
    REQUIRE(g.tensor_graph().num_data() == 3);
    REQUIRE(g.tensor_graph().num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "CONCAT");
    REQUIRE(out->has_producer());
    REQUIRE_FALSE(out->is_leaf());
    REQUIRE(g.num_ops() == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph concat no_grad does not register autograd op", "[graph][nn_graph]")
{
    NNGraph g("nn_concat_nograd");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32);
    auto* b = g.tensor({2, 4}, "b", DataType::FP32);
    NNGraph::TensorNode* out = nullptr;
    {
        auto guard = g.no_grad();
        out = concat(a, b, 1, "out");
    }
    REQUIRE(out != nullptr);
    REQUIRE_FALSE(out->has_producer());
    REQUIRE(g.num_ops() == 0);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph concat rejects invalid arguments", "[graph][nn_graph]")
{
    NNGraph g("nn_concat_bad");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32);
    auto* b = g.tensor({2, 3}, "b", DataType::FP32);

    REQUIRE_THROWS_AS(concat(nullptr, b, 1, "o"), std::invalid_argument);
    REQUIRE_THROWS_AS(concat(a, nullptr, 1, "o"), std::invalid_argument);

    auto* b_bad = g.tensor({3, 3}, "b2", DataType::FP32);
    REQUIRE_THROWS_AS(concat(a, b_bad, 1, "o"), std::invalid_argument);

    NNGraph other("other");
    auto* bo = other.tensor({2, 3}, "bo", DataType::FP32);
    REQUIRE_THROWS_AS(concat(a, bo, 1, "o"), std::invalid_argument);

    auto* bf = g.tensor({2, 3}, "bf64", DataType::FP64);
    REQUIRE_THROWS_AS(concat(a, bf, 1, "o"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph concat forward matches Fortran reference (untiled)", "[graph][nn_graph]")
{
    using ShapesAxis =
        std::tuple<std::vector<Index>, std::vector<Index>, Index>;
    const ShapesAxis c = GENERATE(ShapesAxis{{5}, {3}, 0},
        ShapesAxis{{2, 4}, {2, 5}, 1},
        ShapesAxis{{3, 2}, {4, 2}, 0});

    const auto& a_shape = std::get<0>(c);
    const auto& b_shape = std::get<1>(c);
    const Index axis = std::get<2>(c);

    NNGraph g("nn_concat_ref");
    auto* a = g.tensor(a_shape, "a", DataType::FP32);
    auto* b = g.tensor(b_shape, "b", DataType::FP32);
    a->mark_input(true);
    b->mark_input(true);
    auto* out = concat(a, b, axis, "out");
    out->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    const Index na = shape_prod(a_shape);
    const Index nb = shape_prod(b_shape);
    std::vector<float> a_data(static_cast<size_t>(na));
    std::vector<float> b_data(static_cast<size_t>(nb));
    for(Index i = 0; i < na; ++i)
    {
        a_data[static_cast<size_t>(i)] = static_cast<float>(i + 11);
    }
    for(Index i = 0; i < nb; ++i)
    {
        b_data[static_cast<size_t>(i)] = static_cast<float>(200 + i);
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
    "NNGraph concat tiled matches untiled", "[graph][nn_graph]")
{
    const std::vector<Index> a_shape = {4, 6};
    const std::vector<Index> b_shape = {4, 5};
    constexpr Index axis = 1;

    std::vector<float> a_data(static_cast<size_t>(shape_prod(a_shape)));
    std::vector<float> b_data(static_cast<size_t>(shape_prod(b_shape)));
    for(size_t i = 0; i < a_data.size(); ++i)
    {
        a_data[i] = static_cast<float>(static_cast<int>(i) - 2);
    }
    for(size_t i = 0; i < b_data.size(); ++i)
    {
        b_data[i] = static_cast<float>(40 + i);
    }

    std::vector<float> untiled_result;
    {
        NNGraph g("nn_concat_untiled");
        auto* a = g.tensor(a_shape, "a", DataType::FP32);
        auto* b = g.tensor(b_shape, "b", DataType::FP32);
        a->mark_input(true);
        b->mark_input(true);
        auto* out = concat(a, b, axis, "out");
        out->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
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
        NNGraph g("nn_concat_tiled");
        auto* a = g.tensor(a_shape, "a", DataType::FP32);
        auto* b = g.tensor(b_shape, "b", DataType::FP32);
        a->mark_input(true);
        b->mark_input(true);
        auto* out = concat(a, b, axis, "out");
        out->mark_output(true);

        for(auto* ag : g.tensor_graph().axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph concat backward is not supported yet", "[graph][nn_graph]")
{
    NNGraph g("nn_concat_bwd");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32);
    auto* b = g.tensor({2, 4}, "b", DataType::FP32);
    auto* out = concat(a, b, 1, "out");

    REQUIRE(out->has_producer());
    auto [gout, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(1.0f, gout->data());

    REQUIRE_THROWS_AS(out->backward(), std::runtime_error);
}
