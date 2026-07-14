/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/adamw_step.cc
 * Test TensorGraph adamw_step operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/adamw_step.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/adamw_step.hh"
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

TEST_CASE("TensorGraph adamw_step structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef grad = graph.data({dim_4, dim_5});
    grad->set_name("grad");
    nntile::TensorRef first_moment = graph.data({dim_4, dim_5});
    first_moment->set_name("first_moment");
    nntile::TensorRef second_moment = graph.data({dim_4, dim_5});
    second_moment->set_name("second_moment");
    nntile::TensorRef p = graph.data({dim_4, dim_5});
    p->set_name("p");

    gt::adamw_step(1,
        0.9,
        0.999,
        1e-8,
        0.001,
        0.01,
        grad,
        first_moment,
        second_moment,
        p);

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADAMW_STEP");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 3);
}

TEST_CASE("TensorGraph adamw_step rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef grad = graph.data({5, 4});
    grad->set_name("grad");
    nntile::TensorRef first_moment = graph.data({5, 4});
    first_moment->set_name("first_moment");
    nntile::TensorRef second_moment = graph.data({5, 4});
    second_moment->set_name("second_moment");
    nntile::TensorRef p = graph.data({5, 4});
    p->set_name("p");

    REQUIRE_THROWS_AS(gt::adamw_step(1,
                          0.9,
                          0.999,
                          1e-8,
                          0.001,
                          0.01,
                          nullptr,
                          first_moment,
                          second_moment,
                          p),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::adamw_step(
            1, 0.9, 0.999, 1e-8, 0.001, 0.01, grad, nullptr, second_moment, p),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph adamw_step tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, num_iter, beta_1, beta_2, eps, lr, weight_decay] =
        GENERATE(std::tuple{std::vector<Index>{4, 6},
                     Index(1),
                     0.9,
                     0.999,
                     1e-8,
                     0.001,
                     0.01},
            std::tuple{std::vector<Index>{2, 4},
                Index(2),
                0.95,
                0.99,
                1e-6,
                0.001,
                0.001});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> grad_data(nelems);
    std::vector<float> first_moment_data(nelems);
    std::vector<float> second_moment_data(nelems);
    std::vector<float> p_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        first_moment_data[i] = 0.01f * static_cast<float>(i);
        second_moment_data[i] = 0.02f * static_cast<float>(i + 1);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_first, untiled_second, untiled_p;
    {
        TensorGraph graph("adamw_step_untiled");
        nntile::TensorRef grad_node = graph.data(shape, DataType::FP32);
    grad_node->set_name("grad");
        nntile::TensorRef first_moment_node = graph.data(shape, DataType::FP32);
    first_moment_node->set_name("first_moment");
        nntile::TensorRef second_moment_node = graph.data(shape, DataType::FP32);
    second_moment_node->set_name("second_moment");
        nntile::TensorRef p_node = graph.data(shape, DataType::FP32);
    p_node->set_name("p");

        gt::adamw_step(num_iter,
            beta_1,
            beta_2,
            eps,
            lr,
            weight_decay,
            grad_node,
            first_moment_node,
            second_moment_node,
            p_node);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(grad_node, grad_data);
        runtime.bind_data(first_moment_node, first_moment_data);
        runtime.bind_data(second_moment_node, second_moment_data);
        runtime.bind_data(p_node, p_data);
        runtime.execute();
        runtime.wait();

        untiled_first = runtime.get_output<float>(first_moment_node);
        untiled_second = runtime.get_output<float>(second_moment_node);
        untiled_p = runtime.get_output<float>(p_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_first, tiled_second, tiled_p;
    {
        TensorGraph graph("adamw_step_tiled");
        nntile::TensorRef grad_node = graph.data(shape, DataType::FP32);
    grad_node->set_name("grad");
        nntile::TensorRef first_moment_node = graph.data(shape, DataType::FP32);
    first_moment_node->set_name("first_moment");
        nntile::TensorRef second_moment_node = graph.data(shape, DataType::FP32);
    second_moment_node->set_name("second_moment");
        nntile::TensorRef p_node = graph.data(shape, DataType::FP32);
    p_node->set_name("p");

        gt::adamw_step(num_iter,
            beta_1,
            beta_2,
            eps,
            lr,
            weight_decay,
            grad_node,
            first_moment_node,
            second_moment_node,
            p_node);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(grad_node, grad_data);
        runtime.bind_data(first_moment_node, first_moment_data);
        runtime.bind_data(second_moment_node, second_moment_data);
        runtime.bind_data(p_node, p_data);
        runtime.execute();
        runtime.wait();

        tiled_first = runtime.get_output<float>(first_moment_node);
        tiled_second = runtime.get_output<float>(second_moment_node);
        tiled_p = runtime.get_output<float>(p_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_first.size() == untiled_first.size());
    REQUIRE(tiled_second.size() == untiled_second.size());
    REQUIRE(tiled_p.size() == untiled_p.size());
    for (size_t i = 0; i < tiled_p.size(); ++i)
    {
        REQUIRE(std::abs(tiled_first[i] - untiled_first[i]) < tol);
        REQUIRE(std::abs(tiled_second[i] - untiled_second[i]) < tol);
        REQUIRE(std::abs(tiled_p[i] - untiled_p[i]) < tol);
    }
}
