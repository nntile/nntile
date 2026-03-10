/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/adam_step.cc
 * Test TensorGraph adam_step operation against nntile::tensor::adam_step.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/adam_step.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/adam_step.hh"
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
void check_adam_step_vs_tensor_api(
    const std::vector<Index>& shape,
    Index num_iter,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("adam_step_test");
    auto* grad_node = graph.data(shape, "grad", DataType::FP32);
    auto* first_moment_node = graph.data(shape, "first_moment", DataType::FP32);
    auto* second_moment_node = graph.data(shape, "second_moment", DataType::FP32);
    auto* p_node = graph.data(shape, "p", DataType::FP32);
    grad_node->mark_input(true);
    first_moment_node->mark_input(true);
    second_moment_node->mark_input(true);
    p_node->mark_input(true);
    first_moment_node->mark_output(true);
    second_moment_node->mark_output(true);
    p_node->mark_output(true);

    gt::adam_step(num_iter, beta_1, beta_2, eps, lr, weight_decay,
              grad_node, first_moment_node, second_moment_node, p_node);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> grad_data(nelems);
    std::vector<float> first_moment_data(nelems);
    std::vector<float> second_moment_data(nelems);
    std::vector<float> p_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        first_moment_data[i] = 0.01f * static_cast<float>(i);
        second_moment_data[i] = 0.02f * static_cast<float>(i + 1);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    runtime.bind_data("grad", grad_data);
    runtime.bind_data("first_moment", first_moment_data);
    runtime.bind_data("second_moment", second_moment_data);
    runtime.bind_data("p", p_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_first = runtime.get_output<float>("first_moment");
    std::vector<float> graph_second = runtime.get_output<float>("second_moment");
    std::vector<float> graph_p = runtime.get_output<float>("p");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> grad_t(traits, distr);
    nntile::tensor::Tensor<T> first_moment_t(traits, distr);
    nntile::tensor::Tensor<T> second_moment_t(traits, distr);
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
    init_tile(first_moment_t, first_moment_data);
    init_tile(second_moment_t, second_moment_data);
    init_tile(p_t, p_data);

    nntile::tensor::adam_step<T>(num_iter, beta_1, beta_2, eps, lr, weight_decay,
                         grad_t, first_moment_t, second_moment_t, p_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_first(nelems);
    std::vector<float> tensor_second(nelems);
    std::vector<float> tensor_p(nelems);
    {
        auto tile_m = first_moment_t.get_tile(0);
        auto tile_v = second_moment_t.get_tile(0);
        auto tile_p = p_t.get_tile(0);
        auto loc_m = tile_m.acquire(STARPU_R);
        auto loc_v = tile_v.acquire(STARPU_R);
        auto loc_p = tile_p.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_first[i] = static_cast<float>(loc_m[i]);
            tensor_second[i] = static_cast<float>(loc_v[i]);
            tensor_p[i] = static_cast<float>(loc_p[i]);
        }
        loc_m.release();
        loc_v.release();
        loc_p.release();
    }

    REQUIRE(graph_first.size() == tensor_first.size());
    REQUIRE(graph_second.size() == tensor_second.size());
    REQUIRE(graph_p.size() == tensor_p.size());
    for(size_t i = 0; i < graph_p.size(); ++i)
    {
        REQUIRE(std::abs(graph_first[i] - tensor_first[i]) < tolerance);
        REQUIRE(std::abs(graph_second[i] - tensor_second[i]) < tolerance);
        REQUIRE(std::abs(graph_p[i] - tensor_p[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph adam_step structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* grad = graph.data({dim_4, dim_5}, "grad");
    auto* first_moment = graph.data({dim_4, dim_5}, "first_moment");
    auto* second_moment = graph.data({dim_4, dim_5}, "second_moment");
    auto* p = graph.data({dim_4, dim_5}, "p");

    gt::adam_step(1, 0.9, 0.999, 1e-8, 0.001, 0.0,
             grad, first_moment, second_moment, p);

    REQUIRE(graph.num_data() == 4);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADAM_STEP");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 3);
}

TEST_CASE("TensorGraph adam_step rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* grad = graph.data({4, 5}, "grad");
    auto* first_moment = graph.data({4, 5}, "first_moment");
    auto* second_moment = graph.data({4, 5}, "second_moment");
    auto* p = graph.data({4, 5}, "p");

    REQUIRE_THROWS_AS(
        gt::adam_step(1, 0.9, 0.999, 1e-8, 0.001, 0.0,
                  nullptr, first_moment, second_moment, p),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::adam_step(1, 0.9, 0.999, 1e-8, 0.001, 0.0,
                  grad, nullptr, second_moment, p),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph adam_step matches nntile::tensor::adam_step", "[graph][tensor]")
{
    const auto [shape, num_iter, beta_1, beta_2, eps, lr, weight_decay] =
        GENERATE(
            std::tuple{std::vector<Index>{dim_4, dim_5}, Index(1), 0.9, 0.999,
                       1e-8, 0.001, 0.0},
            std::tuple{std::vector<Index>{6}, Index(2), 0.9, 0.999,
                       1e-8, 0.01, 0.01},
            std::tuple{std::vector<Index>{2, 3}, Index(1), 0.95, 0.99,
                       1e-6, 0.001, 0.0});

    check_adam_step_vs_tensor_api<nntile::fp32_t>(
        shape, num_iter, beta_1, beta_2, eps, lr, weight_decay);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph adam_step tiled matches untiled", "[graph][tensor]")
{
    const auto [shape, num_iter, beta_1, beta_2, eps, lr, weight_decay] =
        GENERATE(
            std::tuple{std::vector<Index>{4, 6}, Index(1), 0.9, 0.999,
                       1e-8, 0.001, 0.0},
            std::tuple{std::vector<Index>{2, 4}, Index(2), 0.95, 0.99,
                       1e-6, 0.001, 0.01});

    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> grad_data(nelems);
    std::vector<float> first_moment_data(nelems);
    std::vector<float> second_moment_data(nelems);
    std::vector<float> p_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        grad_data[i] = 0.1f * static_cast<float>(i + 1);
        first_moment_data[i] = 0.01f * static_cast<float>(i);
        second_moment_data[i] = 0.02f * static_cast<float>(i + 1);
        p_data[i] = 1.0f * static_cast<float>(i - nelems / 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_first, untiled_second, untiled_p;
    {
        TensorGraph graph("adam_step_untiled");
        auto* grad_node = graph.data(shape, "grad", DataType::FP32);
        auto* first_moment_node = graph.data(shape, "first_moment", DataType::FP32);
        auto* second_moment_node = graph.data(shape, "second_moment", DataType::FP32);
        auto* p_node = graph.data(shape, "p", DataType::FP32);
        grad_node->mark_input(true);
        first_moment_node->mark_input(true);
        second_moment_node->mark_input(true);
        p_node->mark_input(true);
        first_moment_node->mark_output(true);
        second_moment_node->mark_output(true);
        p_node->mark_output(true);

        gt::adam_step(num_iter, beta_1, beta_2, eps, lr, weight_decay,
                  grad_node, first_moment_node, second_moment_node, p_node);

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("grad", grad_data);
        runtime.bind_data("first_moment", first_moment_data);
        runtime.bind_data("second_moment", second_moment_data);
        runtime.bind_data("p", p_data);
        runtime.execute();
        runtime.wait();

        untiled_first = runtime.get_output<float>("first_moment");
        untiled_second = runtime.get_output<float>("second_moment");
        untiled_p = runtime.get_output<float>("p");
    }

    // --- Tiled run ---
    std::vector<float> tiled_first, tiled_second, tiled_p;
    {
        TensorGraph graph("adam_step_tiled");
        auto* grad_node = graph.data(shape, "grad", DataType::FP32);
        auto* first_moment_node = graph.data(shape, "first_moment", DataType::FP32);
        auto* second_moment_node = graph.data(shape, "second_moment", DataType::FP32);
        auto* p_node = graph.data(shape, "p", DataType::FP32);
        grad_node->mark_input(true);
        first_moment_node->mark_input(true);
        second_moment_node->mark_input(true);
        p_node->mark_input(true);
        first_moment_node->mark_output(true);
        second_moment_node->mark_output(true);
        p_node->mark_output(true);

        gt::adam_step(num_iter, beta_1, beta_2, eps, lr, weight_decay,
                  grad_node, first_moment_node, second_moment_node, p_node);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("grad", grad_data);
        runtime.bind_data("first_moment", first_moment_data);
        runtime.bind_data("second_moment", second_moment_data);
        runtime.bind_data("p", p_data);
        runtime.execute();
        runtime.wait();

        tiled_first = runtime.get_output<float>("first_moment");
        tiled_second = runtime.get_output<float>("second_moment");
        tiled_p = runtime.get_output<float>("p");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_first.size() == untiled_first.size());
    REQUIRE(tiled_second.size() == untiled_second.size());
    REQUIRE(tiled_p.size() == untiled_p.size());
    for(size_t i = 0; i < tiled_p.size(); ++i)
    {
        REQUIRE(std::abs(tiled_first[i] - untiled_first[i]) < tol);
        REQUIRE(std::abs(tiled_second[i] - untiled_second[i]) < tol);
        REQUIRE(std::abs(tiled_p[i] - untiled_p[i]) < tol);
    }
}
