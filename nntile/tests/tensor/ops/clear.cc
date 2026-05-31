/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/clear.cc
 * Test TensorGraph clear operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/clear.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph clear structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *src = graph.data({dim0, dim1})->set_name("src");

    gt::clear(src);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CLEAR");
    REQUIRE(ops[0]->inputs().size() == 0);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == src);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph clear tiled matches untiled",
    "[graph][tensor]")
{
    const auto shape = GENERATE(std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> init_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        init_data[i] = static_cast<float>(Y(i + 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("clear_untiled");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::clear(dst_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

        runtime.bind_data(dst_node, init_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("clear_tiled");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::clear(dst_node);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

        runtime.bind_data(dst_node, init_data);
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
