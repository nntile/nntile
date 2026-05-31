/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/transpose.cc
 * Test TensorGraph transpose operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/transpose.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/transpose.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha = 1.0;
constexpr Index ndim = 1;

} 

TEST_CASE("TensorGraph transpose structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *src = graph.data({dim0, dim1})->set_name("src");

    auto *dst = gt::transpose(alpha, src, ndim);

    dst->set_name("dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == dim1);
    REQUIRE(dst->shape()[1] == dim0);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "TRANSPOSE");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph transpose rejects duplicate tensors", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;
    TensorGraph graph("test");
    auto *src = graph.data({dim0, dim1})->set_name("src");

    REQUIRE_THROWS_AS(
        gt::transpose(alpha, src, src, Index(1)), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph transpose tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, ndim_val] =
        GENERATE(std::tuple{std::vector<Index>{4, 6}, Index(1)},
            std::tuple{std::vector<Index>{2, 4, 6}, Index(1)});

    using Y = nntile::fp32_t::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("transpose_untiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);

        auto *dst_node = gt::transpose(alpha, src_node, ndim_val);

        dst_node->set_name("dst");
        dst_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("transpose_tiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);

        auto *dst_node = gt::transpose(alpha, src_node, ndim_val);

        dst_node->set_name("dst");
        dst_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

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
