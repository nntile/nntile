/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/hypot_scalar_inverse.cc
 * Test TensorGraph hypot_scalar_inverse operation against nntile::tensor::hypot_scalar_inverse.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/hypot_scalar_inverse.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot_scalar_inverse.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar eps_default = 1e-6;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_half = 0.5;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_hypot_scalar_inverse_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar eps,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("hypot_scalar_inverse_test");
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::hypot_scalar_inverse(eps, alpha, dst_node);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> dst(traits, distr);

    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::hypot_scalar_inverse<T>(eps, alpha, dst);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph hypot_scalar_inverse structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* dst = graph.data({dim0, dim1}, "dst");

    gt::hypot_scalar_inverse(eps_default, alpha_one, dst);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "HYPOT_SCALAR_INVERSE");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph hypot_scalar_inverse matches nntile::tensor::hypot_scalar_inverse",
    "[graph][tensor]")
{
    const auto [eps, alpha, shape] = GENERATE(
        std::tuple{1e-6, 1.0, std::vector<Index>{4, 5}},
        std::tuple{1e-5, 0.5, std::vector<Index>{6}},
        std::tuple{1e-4, 1.0, std::vector<Index>{2, 3}});

    check_hypot_scalar_inverse_vs_tensor_api<nntile::fp32_t>(shape, eps, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph hypot_scalar_inverse tiled matches untiled", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    constexpr Scalar eps = 1e-6;
    constexpr Scalar alpha = 1.0;

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(i + 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("hypot_scalar_inverse_untiled");
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::hypot_scalar_inverse(eps, alpha, dst_node);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("hypot_scalar_inverse_tiled");
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        gt::hypot_scalar_inverse(eps, alpha, dst_node);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("dst");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
