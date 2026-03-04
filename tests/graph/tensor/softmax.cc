/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/softmax.cc
 * Test TensorGraph softmax operation against nntile::tensor::softmax.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/softmax.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/softmax.hh"
#include "nntile/tensor/tensor.hh"

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
void check_softmax_vs_tensor_api(
    const std::vector<Index>& src_shape,
    Index axis,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());
    const std::vector<Index> maxsumexp_shape =
        maxsumexp_dst_shape(src_shape, axis);

    // --- TensorGraph path: src -> maxsumexp -> softmax ---
    TensorGraph graph("softmax_test");
    auto* src_node = graph.data(src_shape, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* maxsumexp_node = gt::maxsumexp(src_node, "maxsumexp", axis, redux);
    auto* dst_node = gt::softmax(maxsumexp_node, src_node, "dst", alpha, axis);
    dst_node->mark_output(true);

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

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_shape, src_shape);
    nntile::tensor::TensorTraits maxsumexp_traits(maxsumexp_shape, maxsumexp_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> mse_distr(maxsumexp_traits.grid.nelems,
                               distr_rank_single);

    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> maxsumexp_t(maxsumexp_traits, mse_distr);
    nntile::tensor::Tensor<T> dst_t(src_traits, src_distr);

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
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }
    nntile::tensor::clear<T>(maxsumexp_t);
    nntile::tensor::maxsumexp<T>(src_t, maxsumexp_t, axis, redux);
    nntile::tensor::softmax<T>(maxsumexp_t, src_t, alpha, dst_t, axis);
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

TEST_CASE("TensorGraph softmax structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* maxsumexp_node = graph.data({2, dim0, dim1}, "maxsumexp");
    auto* src = graph.data({dim0, dim1}, "src");
    auto* dst = gt::softmax(maxsumexp_node, src, "dst", alpha_one, axis_0);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().size() == 2);
    REQUIRE(dst->shape()[0] == dim0);
    REQUIRE(dst->shape()[1] == dim1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SOFTMAX");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph softmax rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* mse = graph.data({2, 4, 5}, "mse");
    auto* src = graph.data({4, 5}, "src");

    REQUIRE_THROWS_AS(
        gt::softmax(nullptr, src, "dst", alpha_one, axis_0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::softmax(mse, nullptr, "dst", alpha_one, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph softmax matches nntile::tensor::softmax", "[graph][tensor]")
{
    const auto [shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, Index(0), 1.0},
        std::tuple{std::vector<Index>{6}, Index(0), 1.0},
        std::tuple{std::vector<Index>{3, 4}, Index(0), 0.5});

    check_softmax_vs_tensor_api<nntile::fp32_t>(shape, axis, alpha);
}
