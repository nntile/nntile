/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/add_slice.cc
 * Test TensorGraph add_slice operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/add_slice.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/add_slice.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Slice shape for add_slice: dst shape with axis removed
static std::vector<Index> slice_shape(
    const std::vector<Index> &dst_shape, Index axis)
{
    std::vector<Index> out;
    out.reserve(dst_shape.size() - 1);
    for (Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if (i != axis)
        {
            out.push_back(dst_shape[i]);
        }
    }
    return out;
}

TEST_CASE("TensorGraph add_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *src1 = graph.data({dim_2})->set_name(
        "src1"); // slice for axis=1: {2,4} without dim 1 = {2}
    auto *src2 = graph.data({dim_4, dim_2})->set_name("src2");

    auto *out = gt::add_slice(alpha_one, src1, beta_one, src2, axis_0);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape() == (std::vector<Index>{dim_4, dim_2}));

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD_SLICE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE("TensorGraph add_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *src1 = graph.data({dim_2})->set_name("src1");
    auto *src2 = graph.data({dim_4, dim_2})->set_name("src2");

    REQUIRE_THROWS_AS(
        gt::add_slice(alpha_one, src1, beta_one, src2, src2, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_slice tiled matches untiled",
    "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha, beta] =
        GENERATE(std::tuple{std::vector<Index>{4, 2}, Index(1), 1.0, 1.0},
            std::tuple{std::vector<Index>{6, 4, 2}, Index(1), 2.0, 0.5});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> src1_sh = slice_shape(dst_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());
    const Index src1_nelems = std::accumulate(
        src1_sh.begin(), src1_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> src1_data(src1_nelems);
    std::vector<float> src2_data(dst_nelems);
    for (Index i = 0; i < src1_nelems; ++i)
        src1_data[i] = static_cast<float>(Y(i + 1));
    for (Index i = 0; i < dst_nelems; ++i)
        src2_data[i] = static_cast<float>(Y(-i - 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("add_slice_untiled");
        auto *src1_node =
            graph.data(src1_sh, DataType::FP32)->set_name("src1");
        auto *src2_node =
            graph.data(dst_shape, DataType::FP32)->set_name("src2");
        src1_node->mark_input(true);
        src2_node->mark_input(true);
        auto *out_node =
            gt::add_slice(alpha, src1_node, beta, src2_node, axis);
        out_node->mark_output(true);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src1_node, src1_data);
        runtime.bind_data(src2_node, src2_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(out_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("add_slice_tiled");
        auto *src1_node =
            graph.data(src1_sh, DataType::FP32)->set_name("src1");
        auto *src2_node =
            graph.data(dst_shape, DataType::FP32)->set_name("src2");
        src1_node->mark_input(true);
        src2_node->mark_input(true);
        auto *out_node =
            gt::add_slice(alpha, src1_node, beta, src2_node, axis);
        out_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src1_node, src1_data);
        runtime.bind_data(src2_node, src2_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(out_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}