/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/softmax_inplace.cc
 * Test TensorGraph softmax_inplace operation against nntile::tensor::softmax_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/softmax_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/softmax_inplace.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr int redux = 0;
constexpr Scalar alpha_one = 1.0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

static std::vector<Index> maxsumexp_dst_shape(
    const std::vector<Index>& src_shape,
    Index axis)
{
    std::vector<Index> dst = {2};
    for(Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if(i != axis)
        {
            dst.push_back(src_shape[i]);
        }
    }
    return dst;
}

template<typename T>
void check_softmax_inplace_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());
    const std::vector<Index> maxsumexp_shape =
        maxsumexp_dst_shape(src_shape, axis);

    // --- TensorGraph path: src -> maxsumexp, dst in-place softmax ---
    TensorGraph graph("softmax_inplace_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    auto* dst_node = graph.data(src_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, redux);
    gt::softmax_inplace(maxsumexp_node, dst_node, alpha, axis);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10 - 2));
        dst_data[i] = src_data[i];
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits maxsumexp_traits(maxsumexp_shape, maxsumexp_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> mse_distr(maxsumexp_traits.grid.nelems,
                               distr_rank_single);
    std::vector<int> dst_distr(src_traits.grid.nelems, distr_rank_single);

    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> maxsumexp_t(maxsumexp_traits, mse_distr);
    nntile::tensor::Tensor<T> dst_t(src_traits, dst_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }
    nntile::tensor::clear<T>(maxsumexp_t);
    nntile::tensor::maxsumexp<T>(src_t, maxsumexp_t, axis, redux);
    nntile::tensor::softmax_inplace<T>(maxsumexp_t, alpha, dst_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(src_nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < src_nelems; ++i)
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

TEST_CASE("TensorGraph softmax_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* maxsumexp_node = graph.data({2, dim0, dim1}, "maxsumexp");
    auto* dst = graph.data({dim0, dim1}, "dst");

    gt::softmax_inplace(maxsumexp_node, dst, alpha_one, axis_0);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SOFTMAX_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph softmax_inplace rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* mse = graph.data({2, 4, 5}, "mse");
    auto* dst = graph.data({4, 5}, "dst");

    REQUIRE_THROWS_AS(
        gt::softmax_inplace(nullptr, dst, alpha_one, axis_0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::softmax_inplace(mse, nullptr, alpha_one, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph softmax_inplace matches nntile::tensor::softmax_inplace",
    "[graph][tensor]")
{
    const auto [shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, Index(0), 1.0},
        std::tuple{std::vector<Index>{6}, Index(0), 1.0},
        std::tuple{std::vector<Index>{3, 4}, Index(0), 0.5});

    check_softmax_inplace_vs_tensor_api<nntile::fp32_t>(shape, axis, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph softmax_inplace tiled matches untiled", "[graph][tensor]")
{
    const auto [shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{4, 6}, Index(0), 1.0},
        std::tuple{std::vector<Index>{3, 4}, Index(0), 0.5});

    using Y = nntile::fp32_t::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10 - 2));
        dst_data[i] = src_data[i];
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("softmax_inplace_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, redux);
        gt::softmax_inplace(maxsumexp_node, dst_node, alpha, axis);

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("dst");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("softmax_inplace_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        auto* dst_node = graph.data(shape, "dst", DataType::FP32);
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, redux);
        gt::softmax_inplace(maxsumexp_node, dst_node, alpha, axis);
        auto* maxsumexp_dim0 = maxsumexp_node->axis(0);
        for(auto* ag : graph.axis_groups())
        {
            if(ag == maxsumexp_dim0)
            {
                ag->set_tiling(ag->extent);
            }
            else
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.bind_data("dst", dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("dst");
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
