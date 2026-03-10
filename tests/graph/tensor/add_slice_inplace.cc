/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/add_slice_inplace.cc
 * Test TensorGraph add_slice_inplace operation against nntile::tensor::add_slice_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/add_slice_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_slice_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_one = 1.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Slice shape: dst shape with axis removed
static std::vector<Index> slice_shape(
    const std::vector<Index>& dst_shape,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(dst_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if(i != axis)
        {
            out.push_back(dst_shape[i]);
        }
    }
    return out;
}

template<typename T>
void check_add_slice_inplace_vs_tensor_api(
    const std::vector<Index>& dst_shape,
    Index axis,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("add_slice_inplace_test");
    auto* src_node = graph.data(src_sh, "src", DataType::FP32);
    auto* dst_node = graph.data(dst_shape, "dst", DataType::FP32);
    src_node->mark_input(true);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    gt::add_slice_inplace(alpha, src_node, beta, dst_node, axis);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }
    for(Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("src", src_data);
    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_sh, src_sh);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> dst_t(dst_traits, dst_distr);

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
        for(Index i = 0; i < dst_nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    nntile::tensor::add_slice_inplace<T>(alpha, src_t, beta, dst_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
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

TEST_CASE("TensorGraph add_slice_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_2}, "src");  // slice for axis=1: {2,4} without dim 1 = {2}
    auto* dst = graph.data({dim_2, dim_4}, "dst");

    gt::add_slice_inplace(alpha_one, src, beta_one, dst, axis_1);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD_SLICE_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph add_slice_inplace rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_4}, "src");

    REQUIRE_THROWS_AS(
        gt::add_slice_inplace(alpha_one, src, beta_one, src, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph add_slice_inplace matches nntile::tensor::add_slice_inplace", "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha, beta] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, alpha_one, beta_zero},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, alpha_one, beta_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, alpha_two, beta_half},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, alpha_one, beta_one});

    check_add_slice_inplace_vs_tensor_api<nntile::fp32_t>(
        dst_shape, axis, alpha, beta);
}
