/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/gemm.cc
 * Test TensorGraph gemm operation against nntile::tensor::gemm.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/constants.hh"
#include "nntile/graph/tensor/gemm.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/gemm.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr bool trans_a = false;
constexpr bool trans_b = false;
constexpr Index ndim = 1;
constexpr Index batch_ndim = 0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_gemm_vs_tensor_api(
    Index M, Index K, Index N,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    std::vector<Index> a_shape = {M, K};
    std::vector<Index> b_shape = {K, N};
    std::vector<Index> c_shape = {M, N};

    const Index a_nelems = M * K;
    const Index b_nelems = K * N;
    const Index c_nelems = M * N;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
    {
        a_data[i] = static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for(Index i = 0; i < b_nelems; ++i)
    {
        b_data[i] = static_cast<float>(Y(i % 7)) * 0.1f;
    }

    // --- TensorGraph path ---
    TensorGraph graph("gemm_test");
    auto* a_node = graph.data(a_shape, "a", DataType::FP32);
    auto* b_node = graph.data(b_shape, "b", DataType::FP32);
    a_node->mark_input(true);
    b_node->mark_input(true);

    auto* c_node = gt::gemm(a_node, b_node, "c", alpha,
                        trans_a, trans_b, ndim, batch_ndim);
    c_node->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("c");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits a_traits(a_shape, a_shape);
    nntile::tensor::TensorTraits b_traits(b_shape, b_shape);
    nntile::tensor::TensorTraits c_traits(c_shape, c_shape);
    std::vector<int> distr_single(1, distr_rank_single);

    nntile::tensor::Tensor<T> a_t(a_traits, distr_single);
    nntile::tensor::Tensor<T> b_t(b_traits, distr_single);
    nntile::tensor::Tensor<T> c_t(c_traits, distr_single);

    {
        auto tile = a_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < a_nelems; ++i)
        {
            loc[i] = static_cast<Y>(a_data[i]);
        }
        loc.release();
    }
    {
        auto tile = b_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < b_nelems; ++i)
        {
            loc[i] = static_cast<Y>(b_data[i]);
        }
        loc.release();
    }
    nntile::tensor::clear<T>(c_t);

    nntile::tensor::gemm<T>(alpha, TransOp(TransOp::NoTrans), a_t,
                    TransOp(TransOp::NoTrans), b_t, 0.0, c_t,
                    ndim, batch_ndim);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(c_nelems);
    {
        auto tile = c_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < c_nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        float diff = std::abs(graph_result[i] - tensor_result[i]);
        float ref = std::abs(tensor_result[i]) + 1e-10f;
        REQUIRE(diff / ref < tolerance);
    }
}

TEST_CASE("TensorGraph gemm structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* a = graph.data({4, 5}, "a");
    auto* b = graph.data({5, 6}, "b");
    auto* c = gt::gemm(a, b, "c", alpha_one, trans_a, trans_b, ndim, batch_ndim);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(c->shape().size() == 2);
    REQUIRE(c->shape()[0] == 4);
    REQUIRE(c->shape()[1] == 6);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "GEMM");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == c);
}

TEST_CASE("TensorGraph gemm rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* a = graph.data({4, 5}, "a");
    auto* b = graph.data({5, 6}, "b");

    REQUIRE_THROWS_AS(
        gt::gemm(nullptr, b, "c", alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::gemm(a, nullptr, "c", alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm matches nntile::tensor::gemm", "[graph][tensor]")
{
    const auto [M, K, N, alpha] = GENERATE(
        std::tuple{Index(4), Index(5), Index(6), 1.0},
        std::tuple{Index(3), Index(4), Index(3), 1.0},
        std::tuple{Index(2), Index(3), Index(4), 0.5});

    check_gemm_vs_tensor_api<nntile::fp32_t>(M, K, N, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm tiled matches untiled", "[graph][tensor]")
{
    const auto [M, K, N, alpha] = GENERATE(
        std::tuple{Index(4), Index(6), Index(8), 1.0},
        std::tuple{Index(2), Index(4), Index(6), 0.5});

    using Y = nntile::fp32_t::repr_t;
    std::vector<Index> a_shape = {M, K};
    std::vector<Index> b_shape = {K, N};

    const Index a_nelems = M * K;
    const Index b_nelems = K * N;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
    {
        a_data[i] = static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for(Index i = 0; i < b_nelems; ++i)
    {
        b_data[i] = static_cast<float>(Y(i % 7)) * 0.1f;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("gemm_untiled");
        auto* a_node = graph.data(a_shape, "a", DataType::FP32);
        auto* b_node = graph.data(b_shape, "b", DataType::FP32);
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto* c_node = gt::gemm(a_node, b_node, "c", alpha,
                            trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("a", a_data);
        runtime.bind_data("b", b_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("c");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("gemm_tiled");
        auto* a_node = graph.data(a_shape, "a", DataType::FP32);
        auto* b_node = graph.data(b_shape, "b", DataType::FP32);
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto* c_node = gt::gemm(a_node, b_node, "c", alpha,
                            trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("a", a_data);
        runtime.bind_data("b", b_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("c");
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
