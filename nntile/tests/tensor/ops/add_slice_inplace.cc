/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/add_slice_inplace.cc
 * Test TensorGraph add_slice_inplace operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/add_slice_inplace.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/add_slice_inplace.hh"
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

//! Slice shape: dst shape with axis removed
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

TEST_CASE("TensorGraph add_slice_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *src = graph.data({dim_2})->set_name(
        "src"); // slice for axis=1: {2,4} without dim 1 = {2}
    auto *dst = graph.data({dim_2, dim_4})->set_name("dst");

    gt::add_slice_inplace(alpha_one, src, beta_one, dst, axis_1);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD_SLICE_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph add_slice_inplace rejects duplicate tensors",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *src = graph.data({dim_4})->set_name("src");

    REQUIRE_THROWS_AS(
        gt::add_slice_inplace(alpha_one, src, beta_one, src, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_slice_inplace tiled matches untiled",
    "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha, beta] =
        GENERATE(std::tuple{std::vector<Index>{2, 4}, Index(1), 1.0, 1.0},
            std::tuple{std::vector<Index>{2, 4, 6}, Index(1), 2.0, 0.5});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for (Index i = 0; i < src_nelems; ++i)
        src_data[i] = static_cast<float>(Y(i + 1));
    for (Index i = 0; i < dst_nelems; ++i)
        dst_data[i] = static_cast<float>(Y(-i - 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("add_slice_inplace_untiled");
        auto *src_node = graph.data(src_sh, DataType::FP32)->set_name("src");
        auto *dst_node =
            graph.data(dst_shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);
        gt::add_slice_inplace(alpha, src_node, beta, dst_node, axis);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(dst_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("add_slice_inplace_tiled");
        auto *src_node = graph.data(src_sh, DataType::FP32)->set_name("src");
        auto *dst_node =
            graph.data(dst_shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);
        gt::add_slice_inplace(alpha, src_node, beta, dst_node, axis);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(dst_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
