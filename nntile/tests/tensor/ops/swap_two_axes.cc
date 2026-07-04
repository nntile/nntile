/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/tensor/ops/swap_two_axes.cc
 * Test TensorGraph swap_two_axes operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/swap_two_axes.hh"

#include "context_fixture.hh"
#include "nntile/core/swap_two_axes_decompose.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

std::vector<float> reference_swap(
    const std::vector<Index> &shape,
    Index dim0,
    Index dim1,
    const std::vector<float> &src_data)
{
    const core::SwapTwoAxesDecomposition decomp =
        core::decompose_swap_axes(shape, dim0, dim1);
    const auto &d = decomp.sizes_5d;
    const Index nelems = static_cast<Index>(src_data.size());
    std::vector<float> dst_data(static_cast<size_t>(nelems));
    for (Index i0 = 0; i0 < d[0]; ++i0)
    {
        for (Index i1 = 0; i1 < d[1]; ++i1)
        {
            for (Index i2 = 0; i2 < d[2]; ++i2)
            {
                for (Index i3 = 0; i3 < d[3]; ++i3)
                {
                    for (Index i4 = 0; i4 < d[4]; ++i4)
                    {
                        const Index src_idx =
                            ((((i0 * d[1] + i1) * d[2] + i2) * d[3] + i3) *
                                d[4] +
                                i4);
                        const Index dst_idx =
                            ((((i0 * d[3] + i3) * d[2] + i2) * d[1] + i1) *
                                d[4] +
                                i4);
                        dst_data[static_cast<size_t>(dst_idx)] =
                            src_data[static_cast<size_t>(src_idx)];
                    }
                }
            }
        }
    }
    return dst_data;
}

} // namespace

TEST_CASE("TensorGraph swap_two_axes structure", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *src = graph.data({4, 5, 6})->set_name("src");
    auto *dst = graph.data({4, 6, 5})->set_name("dst");
    gt::swap_two_axes(src, dst, 1, 2);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == 4);
    REQUIRE(dst->shape()[1] == 6);
    REQUIRE(dst->shape()[2] == 5);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SWAP_TWO_AXES");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph swap_two_axes rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *src = graph.data({5, 4})->set_name("src");
    REQUIRE_THROWS_AS(gt::swap_two_axes(src, src, 0, 1), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph swap_two_axes tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, dim0, dim1] = GENERATE(
        std::tuple{std::vector<Index>{4, 6}, Index(0), Index(1)},
        std::tuple{std::vector<Index>{2, 8, 4, 16}, Index(1), Index(2)},
        std::tuple{std::vector<Index>{2, 4, 6}, Index(0), Index(2)});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());
    std::vector<float> src_data(static_cast<size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[static_cast<size_t>(i)] = static_cast<float>(i * 2 - 3);
    }
    const std::vector<float> expected =
        reference_swap(shape, dim0, dim1, src_data);

    const core::SwapTwoAxesDecomposition decomp =
        core::decompose_swap_axes(shape, dim0, dim1);
    const auto &out_shape = decomp.output_shape;

    std::vector<float> untiled_result;
    {
        TensorGraph graph("swap_two_axes_untiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);
        auto *dst_node = graph.data(out_shape, DataType::FP32)->set_name("dst");
        dst_node->mark_output(true);
        gt::swap_two_axes(src_node, dst_node, dim0, dim1);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(dst_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("swap_two_axes_tiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);
        auto *dst_node = graph.data(out_shape, DataType::FP32)->set_name("dst");
        dst_node->mark_output(true);
        gt::swap_two_axes(src_node, dst_node, dim0, dim1);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(dst_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    REQUIRE(tiled_result.size() == expected.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
        REQUIRE(std::abs(tiled_result[i] - expected[i]) < tol);
    }
}
