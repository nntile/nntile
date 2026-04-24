/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/mask_scalar.cc
 * Test TensorGraph mask_scalar operation against nntile::tensor::mask_scalar.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/mask_scalar.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/mask_scalar.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar val = -0.5;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_mask_scalar_vs_tensor_api(
    const std::vector<Index>& A_shape,
    Index batch_ndim)
{
    using Y = typename T::repr_t;
    const Index A_nelems = std::accumulate(
        A_shape.begin(), A_shape.end(), Index(1), std::multiplies<>());
    const Index A_data_ndim =
        static_cast<Index>(A_shape.size()) - batch_ndim;
    std::vector<Index> mask_shape(A_shape.begin(),
                                  A_shape.begin() + A_data_ndim);
    const Index mask_nelems = std::accumulate(
        mask_shape.begin(), mask_shape.end(), Index(1), std::multiplies<>());

    // Build mask: true = keep, false = replace with val
    std::vector<float> mask_data(mask_nelems);
    std::vector<float> A_data(A_nelems);
    for(Index i = 0; i < mask_nelems; ++i)
    {
        mask_data[i] = (i % 2 == 0) ? 0.0f : 1.0f;  // even -> false, odd -> true
    }
    for(Index i = 0; i < A_nelems; ++i)
    {
        A_data[i] = static_cast<float>(Y(i + 1));
    }

    // --- TensorGraph path ---
    TensorGraph graph("mask_scalar_test");
    auto* mask_node = graph.data(mask_shape, "mask", DataType::BOOL);
    auto* A_node = graph.data(A_shape, "A", DataType::FP32);
    mask_node->mark_input(true);
    A_node->mark_input(true);
    A_node->mark_output(true);

    gt::mask_scalar(mask_node, val, A_node, batch_ndim);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    runtime.bind_data("mask", mask_data);
    runtime.bind_data("A", A_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("A");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits A_traits(A_shape, A_shape);
    std::vector<int> distr(A_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> A_t(A_traits, distr);

    nntile::tensor::TensorTraits mask_traits(mask_shape, mask_shape);
    nntile::tensor::Tensor<nntile::bool_t> mask_t(mask_traits, distr);

    {
        auto tile = A_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < A_nelems; ++i)
        {
            loc[i] = static_cast<Y>(A_data[i]);
        }
        loc.release();
    }
    {
        auto tile = mask_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < mask_nelems; ++i)
        {
            loc[i] = nntile::bool_t(mask_data[i] != 0.0f);
        }
        loc.release();
    }

    nntile::tensor::mask_scalar<T>(mask_t, val, A_t, batch_ndim);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(A_nelems);
    {
        auto tile = A_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < A_nelems; ++i)
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

TEST_CASE("TensorGraph mask_scalar structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;
    constexpr Index batch_ndim = 0;

    TensorGraph graph("test");

    auto* mask = graph.data({dim0, dim1}, "mask", DataType::BOOL);
    auto* A = graph.data({dim0, dim1}, "A");

    gt::mask_scalar(mask, val, A, batch_ndim);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MASK_SCALAR");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == A);
}

TEST_CASE("TensorGraph mask_scalar rejects null tensors", "[graph][tensor]")
{
    constexpr Index batch_ndim = 0;
    TensorGraph graph("test");
    auto* mask = graph.data({4, 5}, "mask", DataType::BOOL);
    auto* A = graph.data({4, 5}, "A");

    REQUIRE_THROWS_AS(
        gt::mask_scalar(nullptr, val, A, batch_ndim),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::mask_scalar(mask, val, nullptr, batch_ndim),
        std::invalid_argument);
}

TEST_CASE("TensorGraph mask_scalar rejects non-BOOL mask", "[graph][tensor]")
{
    constexpr Index batch_ndim = 0;
    TensorGraph graph("test");
    auto* mask = graph.data({4, 5}, "mask");  // FP32 by default
    auto* A = graph.data({4, 5}, "A");

    REQUIRE_THROWS_AS(
        gt::mask_scalar(mask, val, A, batch_ndim),
        std::invalid_argument);
}

TEST_CASE("TensorGraph mask_scalar rejects mismatched mask ndim", "[graph][tensor]")
{
    TensorGraph graph("test");
    // A is seq x seq x batch (3D), mask must be seq x seq (2D)
    auto* mask = graph.data({4, 5, 8}, "mask", DataType::BOOL);  // wrong: 3D
    auto* A = graph.data({4, 5, 8}, "A");
    REQUIRE_THROWS_AS(
        gt::mask_scalar(mask, val, A, 1),
        std::invalid_argument);

    // mask 1D when A_data is 2D
    TensorGraph graph2("test2");
    auto* mask2 = graph2.data({4}, "mask", DataType::BOOL);
    auto* A2 = graph2.data({4, 5}, "A");
    REQUIRE_THROWS_AS(
        gt::mask_scalar(mask2, val, A2, 0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph mask_scalar matches nntile::tensor::mask_scalar", "[graph][tensor]")
{
    const auto [A_shape, batch_ndim] = GENERATE(
        std::make_pair(std::vector<Index>{4, 5}, Index(0)),
        std::make_pair(std::vector<Index>{6}, Index(0)),
        std::make_pair(std::vector<Index>{2, 3}, Index(0)),
        std::make_pair(std::vector<Index>{4, 5, 8}, Index(1)),
        std::make_pair(std::vector<Index>{2, 3, 4, 5}, Index(2)));

    check_mask_scalar_vs_tensor_api<nntile::fp32_t>(A_shape, batch_ndim);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph mask_scalar tiled matches untiled", "[graph][tensor]")
{
    const auto [A_shape, batch_ndim] = GENERATE(
        std::make_pair(std::vector<Index>{4, 6}, Index(0)),
        std::make_pair(std::vector<Index>{6}, Index(0)),
        std::make_pair(std::vector<Index>{2, 4}, Index(0)),
        std::make_pair(std::vector<Index>{4, 6, 8}, Index(1)));

    const Index A_data_ndim =
        static_cast<Index>(A_shape.size()) - batch_ndim;
    std::vector<Index> mask_shape(A_shape.begin(),
                                  A_shape.begin() + A_data_ndim);
    const Index mask_nelems = std::accumulate(
        mask_shape.begin(), mask_shape.end(), Index(1), std::multiplies<>());
    const Index A_nelems = std::accumulate(
        A_shape.begin(), A_shape.end(), Index(1), std::multiplies<>());

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    std::vector<float> mask_data(mask_nelems);
    std::vector<float> A_data(A_nelems);
    for(Index i = 0; i < mask_nelems; ++i)
    {
        mask_data[i] = (i % 2 == 0) ? 0.0f : 1.0f;
    }
    for(Index i = 0; i < A_nelems; ++i)
    {
        A_data[i] = static_cast<float>(Y(i + 1));
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("mask_scalar_untiled");
        auto* mask_node = graph.data(mask_shape, "mask", DataType::BOOL);
        auto* A_node = graph.data(A_shape, "A", DataType::FP32);
        mask_node->mark_input(true);
        A_node->mark_input(true);
        A_node->mark_output(true);

        gt::mask_scalar(mask_node, val, A_node, batch_ndim);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("mask", mask_data);
        runtime.bind_data("A", A_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("A");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("mask_scalar_tiled");
        auto* mask_node = graph.data(mask_shape, "mask", DataType::BOOL);
        auto* A_node = graph.data(A_shape, "A", DataType::FP32);
        mask_node->mark_input(true);
        A_node->mark_input(true);
        A_node->mark_output(true);

        gt::mask_scalar(mask_node, val, A_node, batch_ndim);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data("mask", mask_data);
        runtime.bind_data("A", A_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("A");
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
