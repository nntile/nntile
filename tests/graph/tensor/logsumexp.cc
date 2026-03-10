/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/logsumexp.cc
 * Test TensorGraph logsumexp operation against nntile::tensor::logsumexp.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/logsumexp.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/logsumexp.hh"
#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr int redux = 0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

// maxsumexp output shape: [2] + src.shape without axis
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
void check_logsumexp_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());
    const std::vector<Index> maxsumexp_shape =
        maxsumexp_dst_shape(src_shape, axis);
    const Index maxsumexp_nelems = std::accumulate(
        maxsumexp_shape.begin(), maxsumexp_shape.end(), Index(1),
        std::multiplies<>());
    const std::vector<Index> logsumexp_shape(maxsumexp_shape.begin() + 1,
                                            maxsumexp_shape.end());
    const Index logsumexp_nelems = std::accumulate(
        logsumexp_shape.begin(), logsumexp_shape.end(), Index(1),
        std::multiplies<>());

    // --- TensorGraph path: src -> maxsumexp -> logsumexp ---
    TensorGraph graph("logsumexp_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, redux);
    auto* logsumexp_node = gt::logsumexp(maxsumexp_node, "logsumexp");
    logsumexp_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10 - 2));
    }

    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result =
        runtime.get_output<float>("logsumexp");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits maxsumexp_traits(maxsumexp_shape, maxsumexp_shape);
    nntile::tensor::TensorTraits logsumexp_traits(logsumexp_shape, logsumexp_shape);
    std::vector<int> distr_single(1, distr_rank_single);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> mse_distr(maxsumexp_traits.grid.nelems,
                               distr_rank_single);
    std::vector<int> lse_distr(logsumexp_traits.grid.nelems,
                              distr_rank_single);

    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> maxsumexp_t(maxsumexp_traits, mse_distr);
    nntile::tensor::Tensor<T> logsumexp_t(logsumexp_traits, lse_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    nntile::tensor::clear<T>(maxsumexp_t);
    nntile::tensor::maxsumexp<T>(src_t, maxsumexp_t, axis, redux);
    nntile::tensor::logsumexp<T>(maxsumexp_t, logsumexp_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(logsumexp_nelems);
    {
        auto tile = logsumexp_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < logsumexp_nelems; ++i)
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

TEST_CASE("TensorGraph logsumexp structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({2, 4, 5}, "src");  // maxsumexp output shape
    auto* dst = gt::logsumexp(src, "dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().size() == 2);
    REQUIRE(dst->shape()[0] == 4);
    REQUIRE(dst->shape()[1] == 5);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "LOGSUMEXP");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph logsumexp rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");

    REQUIRE_THROWS_AS(gt::logsumexp(nullptr, "dst"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph logsumexp matches nntile::tensor::logsumexp", "[graph][tensor]")
{
    const auto [shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, Index(0)},
        std::tuple{std::vector<Index>{6}, Index(0)},
        std::tuple{std::vector<Index>{3, 4}, Index(0)});

    check_logsumexp_vs_tensor_api<nntile::fp32_t>(shape, axis);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph logsumexp tiled matches untiled", "[graph][tensor]")
{
    const auto [shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{4, 6}, Index(0)},
        std::tuple{std::vector<Index>{3, 4}, Index(0)});

    const Index src_nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10 - 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("logsumexp_untiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        src_node->mark_input(true);

        auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, 0);
        auto* logsumexp_node = gt::logsumexp(maxsumexp_node, "logsumexp");
        logsumexp_node->mark_output(true);

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>("logsumexp");
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("logsumexp_tiled");
        auto* src_node = graph.data(shape, "src", DataType::FP32);
        src_node->mark_input(true);

        auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, 0);
        auto* logsumexp_node = gt::logsumexp(maxsumexp_node, "logsumexp");
        logsumexp_node->mark_output(true);
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

        TensorGraph::Runtime runtime(graph);
        runtime.compile();

        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>("logsumexp");
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
