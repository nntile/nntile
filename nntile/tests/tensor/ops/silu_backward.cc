/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/silu_backward.cc
 * Test TensorGraph silu_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/silu_backward.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/silu_backward.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph silu_backward structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *x = graph.data({dim0, dim1})->set_name("x");
    auto *dy = graph.data({dim0, dim1})->set_name("dy");

    auto *dx = gt::silu_backward(x, dy)->set_name("dx");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dx->shape()[0] == dim0);
    REQUIRE(dx->shape()[1] == dim1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SILU_BACKWARD");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dx);
}

TEST_CASE(
    "TensorGraph silu_backward rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *x = graph.data({4, 5})->set_name("x");
    auto *dy = graph.data({4, 5})->set_name("dy");

    REQUIRE_THROWS_AS(gt::silu_backward(x, x), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::silu_backward(x, dy, x), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::silu_backward(x, dy, dy), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph silu_backward tiled matches untiled",
    "[graph][tensor]")
{
    const auto shape = GENERATE(std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(nelems), dy_data(nelems), dx_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i - nelems / 2));
        dy_data[i] = static_cast<float>(Y(i % 7 + 1));
        dx_data[i] = 0.0f;
    }

    std::vector<float> untiled_result;
    {
        TensorGraph graph("silu_backward_untiled");
        auto *x_node = graph.data(shape, DataType::FP32)->set_name("x");
        auto *dy_node = graph.data(shape, DataType::FP32)->set_name("dy");
        auto *dx_node = graph.data(shape, DataType::FP32)->set_name("dx");
        x_node->mark_input(true);
        dy_node->mark_input(true);
        dx_node->mark_input(true);
        dx_node->mark_output(true);
        gt::silu_backward(x_node, dy_node, dx_node);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();
        runtime.bind_data(x_node, x_data);
        runtime.bind_data(dy_node, dy_data);
        runtime.bind_data(dx_node, dx_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(dx_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("silu_backward_tiled");
        auto *x_node = graph.data(shape, DataType::FP32)->set_name("x");
        auto *dy_node = graph.data(shape, DataType::FP32)->set_name("dy");
        auto *dx_node = graph.data(shape, DataType::FP32)->set_name("dx");
        x_node->mark_input(true);
        dy_node->mark_input(true);
        dx_node->mark_input(true);
        dx_node->mark_output(true);
        gt::silu_backward(x_node, dy_node, dx_node);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();
        runtime.bind_data(x_node, x_data);
        runtime.bind_data(dy_node, dy_data);
        runtime.bind_data(dx_node, dx_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(dx_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
