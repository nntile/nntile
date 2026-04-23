/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/sgd_step.cc
 * Test TensorGraph sgd_step operation against nntile::tensor::sgd_step.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/sgd_step.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sgd_step.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_sgd_step_vs_tensor_api(
    const std::vector<Index>& shape,
    Index num_iter,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("sgd_step_test");
    auto* grad_node = graph.data(shape, "grad", DataType::FP32);
    auto* velocity_node = graph.data(shape, "velocity", DataType::FP32);
    auto* p_node = graph.data(shape, "p", DataType::FP32);
    grad_node->mark_input(true);
    velocity_node->mark_input(true);
    p_node->mark_input(true);
    velocity_node->mark_output(true);
    p_node->mark_output(true);

    gt::sgd_step(num_iter, momentum, lr, weight_decay, dampening, nesterov,
             grad_node, velocity_node, p_node);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> grad_data(nelems);
    std::vector<float> velocity_data(nelems);
    std::vector<float> p_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        velocity_data[i] = 0.01f * static_cast<float>(i);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    runtime.bind_data("grad", grad_data);
    runtime.bind_data("velocity", velocity_data);
    runtime.bind_data("p", p_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_velocity = runtime.get_output<float>("velocity");
    std::vector<float> graph_p = runtime.get_output<float>("p");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> grad_t(traits, distr);
    nntile::tensor::Tensor<T> velocity_t(traits, distr);
    nntile::tensor::Tensor<T> p_t(traits, distr);

    auto init_tile = [&](nntile::tensor::Tensor<T>& t, const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(data[i]);
        }
        loc.release();
    };
    init_tile(grad_t, grad_data);
    init_tile(velocity_t, velocity_data);
    init_tile(p_t, p_data);

    nntile::tensor::sgd_step<T>(num_iter, momentum, lr, weight_decay, dampening,
                        nesterov, grad_t, velocity_t, p_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_velocity(nelems);
    std::vector<float> tensor_p(nelems);
    {
        auto tile_v = velocity_t.get_tile(0);
        auto tile_p = p_t.get_tile(0);
        auto loc_v = tile_v.acquire(STARPU_R);
        auto loc_p = tile_p.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_velocity[i] = static_cast<float>(loc_v[i]);
            tensor_p[i] = static_cast<float>(loc_p[i]);
        }
        loc_v.release();
        loc_p.release();
    }

    REQUIRE(graph_velocity.size() == tensor_velocity.size());
    REQUIRE(graph_p.size() == tensor_p.size());
    for(size_t i = 0; i < graph_p.size(); ++i)
    {
        REQUIRE(std::abs(graph_velocity[i] - tensor_velocity[i]) < tolerance);
        REQUIRE(std::abs(graph_p[i] - tensor_p[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph sgd_step structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* grad = graph.data({dim_4, dim_5}, "grad");
    auto* velocity = graph.data({dim_4, dim_5}, "velocity");
    auto* p = graph.data({dim_4, dim_5}, "p");

    gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false,
             grad, velocity, p);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SGD_STEP");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 2);
}

TEST_CASE("TensorGraph sgd_step rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* grad = graph.data({4, 5}, "grad");
    auto* velocity = graph.data({4, 5}, "velocity");
    auto* p = graph.data({4, 5}, "p");

    REQUIRE_THROWS_AS(
        gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false,
                 nullptr, velocity, p),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::sgd_step(1, 0.9, 0.001, 0.0, 0.0, false,
                 grad, nullptr, p),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sgd_step matches nntile::tensor::sgd_step", "[graph][tensor]")
{
    const auto [shape, num_iter, momentum, lr, weight_decay, dampening, nesterov] =
        GENERATE(
            std::tuple{std::vector<Index>{dim_4, dim_5}, Index(1), 0.9, 0.001,
                       0.0, 0.0, false},
            std::tuple{std::vector<Index>{6}, Index(2), 0.0, 0.01,
                       0.0, 0.0, false},
            std::tuple{std::vector<Index>{2, 3}, Index(1), 0.9, 0.001,
                       0.01, 0.0, false},
            std::tuple{std::vector<Index>{4, 5}, Index(1), 0.9, 0.001,
                       0.0, 0.0, true});

    check_sgd_step_vs_tensor_api<nntile::fp32_t>(
        shape, num_iter, momentum, lr, weight_decay, dampening, nesterov);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sgd_step tiled matches untiled", "[graph][tensor]")
{
    const auto [shape, num_iter, momentum, lr, weight_decay, dampening, nesterov] =
        GENERATE(
            std::tuple{std::vector<Index>{4, 6}, Index(1), 0.9, 0.001,
                       0.0, 0.0, false},
            std::tuple{std::vector<Index>{2, 4}, Index(1), 0.9, 0.001,
                       0.01, 0.0, false});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> grad_data(nelems);
    std::vector<float> velocity_data(nelems);
    std::vector<float> p_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        velocity_data[i] = 0.01f * static_cast<float>(i);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_velocity, untiled_p;
    {
        TensorGraph graph("sgd_step_untiled");
        auto* grad_node = graph.data(shape, "grad", DataType::FP32);
        auto* velocity_node = graph.data(shape, "velocity", DataType::FP32);
        auto* p_node = graph.data(shape, "p", DataType::FP32);
        grad_node->mark_input(true);
        velocity_node->mark_input(true);
        p_node->mark_input(true);
        velocity_node->mark_output(true);
        p_node->mark_output(true);

        gt::sgd_step(num_iter, momentum, lr, weight_decay, dampening, nesterov,
                 grad_node, velocity_node, p_node);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("grad", grad_data);
        runtime.bind_data("velocity", velocity_data);
        runtime.bind_data("p", p_data);
        runtime.execute();
        runtime.wait();

        untiled_velocity = runtime.get_output<float>("velocity");
        untiled_p = runtime.get_output<float>("p");
    }

    // --- Tiled run ---
    std::vector<float> tiled_velocity, tiled_p;
    {
        TensorGraph graph("sgd_step_tiled");
        auto* grad_node = graph.data(shape, "grad", DataType::FP32);
        auto* velocity_node = graph.data(shape, "velocity", DataType::FP32);
        auto* p_node = graph.data(shape, "p", DataType::FP32);
        grad_node->mark_input(true);
        velocity_node->mark_input(true);
        p_node->mark_input(true);
        velocity_node->mark_output(true);
        p_node->mark_output(true);

        gt::sgd_step(num_iter, momentum, lr, weight_decay, dampening, nesterov,
                 grad_node, velocity_node, p_node);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("grad", grad_data);
        runtime.bind_data("velocity", velocity_data);
        runtime.bind_data("p", p_data);
        runtime.execute();
        runtime.wait();

        tiled_velocity = runtime.get_output<float>("velocity");
        tiled_p = runtime.get_output<float>("p");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_velocity.size() == untiled_velocity.size());
    REQUIRE(tiled_p.size() == untiled_p.size());
    for(size_t i = 0; i < tiled_p.size(); ++i)
    {
        REQUIRE(std::abs(tiled_velocity[i] - untiled_velocity[i]) < tol);
        REQUIRE(std::abs(tiled_p[i] - untiled_p[i]) < tol);
    }
}
