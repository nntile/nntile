/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/multiply.cc
 * Test TensorGraph multiply operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/multiply.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/multiply.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph multiply structure", "[graph][tensor]")
{
    const auto alpha = GENERATE(1.0, 2.5, 0.5);
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *x = graph.data({dim0, dim1})->set_name("x");
    auto *y = graph.data({dim0, dim1})->set_name("y");

    auto *z = gt::multiply(x, y, alpha);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(z->shape()[0] == dim0);
    REQUIRE(z->shape()[1] == dim1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == z);
}

TEST_CASE("TensorGraph multiply rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *x = graph.data({4, 5})->set_name("x");

    REQUIRE_THROWS_AS(gt::multiply(x, x, 1.0), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply tiled matches untiled",
    "[graph][tensor]")
{
    const auto [alpha, shape] =
        GENERATE(std::tuple{1.0, std::vector<Index>{4, 6}},
            std::tuple{2.5, std::vector<Index>{2, 4}},
            std::tuple{0.5, std::vector<Index>{6}});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(nelems), y_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + 1));
        y_data[i] = static_cast<float>(Y(-i - 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("multiply_untiled");
        auto *x_node = graph.data(shape, DataType::FP32)->set_name("x");
        auto *y_node = graph.data(shape, DataType::FP32)->set_name("y");
        x_node->mark_input(true);
        y_node->mark_input(true);

        auto *z_node = gt::multiply(x_node, y_node, alpha);
        z_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(z_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("multiply_tiled");
        auto *x_node = graph.data(shape, DataType::FP32)->set_name("x");
        auto *y_node = graph.data(shape, DataType::FP32)->set_name("y");
        x_node->mark_input(true);
        y_node->mark_input(true);

        auto *z_node = gt::multiply(x_node, y_node, alpha);
        z_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile_with_round_robin_schedule();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(z_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
