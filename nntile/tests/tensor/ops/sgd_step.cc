/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/sgd_step.cc
 * Test TensorGraph sgd_step operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/sgd_step.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/sgd_step.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph sgd_step structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *grad = graph.data({dim_4, dim_5})->set_name("grad");
    auto *velocity = graph.data({dim_4, dim_5})->set_name("velocity");
    auto *p = graph.data({dim_4, dim_5})->set_name("p");

    gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false, grad, velocity, p);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SGD_STEP");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 2);
}

TEST_CASE("TensorGraph sgd_step rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *grad = graph.data({4, 5})->set_name("grad");
    auto *velocity = graph.data({4, 5})->set_name("velocity");
    auto *p = graph.data({4, 5})->set_name("p");

    REQUIRE_THROWS_AS(
        gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false, nullptr, velocity, p),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false, grad, nullptr, p),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sgd_step tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape,
        num_iter,
        momentum,
        lr,
        weight_decay,
        dampening,
        nesterov] = GENERATE(std::tuple{std::vector<Index>{4, 6},
                                 Index(1),
                                 0.9,
                                 0.001,
                                 0.0,
                                 0.0,
                                 false},
        std::tuple{
            std::vector<Index>{2, 4}, Index(1), 0.9, 0.001, 0.01, 0.0, false});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> grad_data(nelems);
    std::vector<float> velocity_data(nelems);
    std::vector<float> p_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        velocity_data[i] = 0.01f * static_cast<float>(i);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_velocity, untiled_p;
    {
        TensorGraph graph("sgd_step_untiled");
        auto *grad_node = graph.data(shape, DataType::FP32)->set_name("grad");
        auto *velocity_node =
            graph.data(shape, DataType::FP32)->set_name("velocity");
        auto *p_node = graph.data(shape, DataType::FP32)->set_name("p");
        grad_node->mark_input(true);
        velocity_node->mark_input(true);
        p_node->mark_input(true);
        velocity_node->mark_output(true);
        p_node->mark_output(true);

        gt::sgd_step(num_iter,
            momentum,
            lr,
            weight_decay,
            dampening,
            nesterov,
            grad_node,
            velocity_node,
            p_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(grad_node, grad_data);
        runtime.bind_data(velocity_node, velocity_data);
        runtime.bind_data(p_node, p_data);
        runtime.execute();
        runtime.wait();

        untiled_velocity = runtime.get_output<float>(velocity_node);
        untiled_p = runtime.get_output<float>(p_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_velocity, tiled_p;
    {
        TensorGraph graph("sgd_step_tiled");
        auto *grad_node = graph.data(shape, DataType::FP32)->set_name("grad");
        auto *velocity_node =
            graph.data(shape, DataType::FP32)->set_name("velocity");
        auto *p_node = graph.data(shape, DataType::FP32)->set_name("p");
        grad_node->mark_input(true);
        velocity_node->mark_input(true);
        p_node->mark_input(true);
        velocity_node->mark_output(true);
        p_node->mark_output(true);

        gt::sgd_step(num_iter,
            momentum,
            lr,
            weight_decay,
            dampening,
            nesterov,
            grad_node,
            velocity_node,
            p_node);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(grad_node, grad_data);
        runtime.bind_data(velocity_node, velocity_data);
        runtime.bind_data(p_node, p_data);
        runtime.execute();
        runtime.wait();

        tiled_velocity = runtime.get_output<float>(velocity_node);
        tiled_p = runtime.get_output<float>(p_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_velocity.size() == untiled_velocity.size());
    REQUIRE(tiled_p.size() == untiled_p.size());
    for (size_t i = 0; i < tiled_p.size(); ++i)
    {
        REQUIRE(std::abs(tiled_velocity[i] - untiled_velocity[i]) < tol);
        REQUIRE(std::abs(tiled_p[i] - untiled_p[i]) < tol);
    }
}
