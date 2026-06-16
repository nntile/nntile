/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/gemm.cc
 * Test TensorGraph gemm operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gemm.hh"

#include "context_fixture.hh"
#include "nntile/constants.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/gemm.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr bool trans_a = false;
constexpr bool trans_b = false;
constexpr Index ndim = 1;
constexpr Index batch_ndim = 0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph gemm structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *a = graph.data({6, 5})->set_name("a");
    auto *b = graph.data({5, 4})->set_name("b");
    auto *c = gt::gemm(a, b, alpha_one, trans_a, trans_b, ndim, batch_ndim);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(c->shape().size() == 2);
    REQUIRE(c->shape()[0] == 6);
    REQUIRE(c->shape()[1] == 4);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "GEMM");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == c);
}

TEST_CASE("TensorGraph gemm rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *a = graph.data({6, 5})->set_name("a");
    auto *b = graph.data({5, 4})->set_name("b");

    REQUIRE_THROWS_AS(
        gt::gemm(nullptr, b, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::gemm(a, nullptr, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm tiled matches untiled",
    "[graph][tensor]")
{
    const auto [M, K, N, alpha] =
        GENERATE(std::tuple{Index(4), Index(6), Index(8), 1.0},
            std::tuple{Index(2), Index(4), Index(6), 0.5});

    using Y = nntile::fp32_t::repr_t;
    std::vector<Index> a_shape = {N, K};
    std::vector<Index> b_shape = {K, M};

    const Index a_nelems = N * K;
    const Index b_nelems = K * M;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for (Index i = 0; i < a_nelems; ++i)
    {
        a_data[i] = static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for (Index i = 0; i < b_nelems; ++i)
    {
        b_data[i] = static_cast<float>(Y(i % 7)) * 0.1f;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("gemm_untiled");
        auto *a_node = graph.data(a_shape, DataType::FP32)->set_name("a");
        auto *b_node = graph.data(b_shape, DataType::FP32)->set_name("b");
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto *c_node = gt::gemm(
            a_node, b_node, alpha, trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(a_node, a_data);
        runtime.bind_data(b_node, b_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(c_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("gemm_tiled");
        auto *a_node = graph.data(a_shape, DataType::FP32)->set_name("a");
        auto *b_node = graph.data(b_shape, DataType::FP32)->set_name("b");
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto *c_node = gt::gemm(
            a_node, b_node, alpha, trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(a_node, a_data);
        runtime.bind_data(b_node, b_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(c_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
