/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/conv2d_bwd_weight_inplace.cc
 * Test TensorGraph conv2d_bwd_weight_inplace operation against
 * nntile::tensor::conv2d_bwd_weight_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <array>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/conv2d_bwd_weight_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/conv2d_bwd_weight_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

// dY shape from X shape (W,H,C_in,N), dC shape (K_x,K_y,C_in,C_out), padding, stride, dilation
std::vector<Index> conv2d_output_shape(
    const std::vector<Index>& x_shape,
    const std::vector<Index>& dc_shape,
    const std::array<Index, 2>& padding,
    const std::array<Index, 2>& stride,
    const std::array<Index, 2>& dilation)
{
    Index W_out = (x_shape[0] + 2 * padding[0] - dilation[0] * (dc_shape[0] - 1) - 1)
                  / stride[0] + 1;
    Index H_out = (x_shape[1] + 2 * padding[1] - dilation[1] * (dc_shape[1] - 1) - 1)
                  / stride[1] + 1;
    return {W_out, H_out, dc_shape[3], x_shape[3]};
}

} // anonymous namespace

TEST_CASE("TensorGraph conv2d_bwd_weight_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* x = graph.data({4, 4, 2, 2}, "x");
    auto* dy = graph.data({3, 3, 2, 2}, "dy");
    auto* dc = graph.data({2, 2, 2, 2}, "dc");

    gt::conv2d_bwd_weight_inplace(1.0, x, dy, 0.0, dc, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CONV2D_BWD_WEIGHT_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dc);
}

TEST_CASE("TensorGraph conv2d_bwd_weight_inplace rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* x = graph.data({4, 4, 2, 2}, "x");
    auto* dy = graph.data({3, 3, 2, 2}, "dy");
    auto* dc = graph.data({2, 2, 2, 2}, "dc");

    REQUIRE_THROWS_AS(
        gt::conv2d_bwd_weight_inplace(1.0, nullptr, dy, 0.0, dc, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::conv2d_bwd_weight_inplace(1.0, x, nullptr, 0.0, dc, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
}

template<typename T>
void check_conv2d_bwd_weight_inplace_vs_tensor_api(
    const std::vector<Index>& x_shape,
    const std::vector<Index>& dc_shape,
    Scalar alpha,
    Scalar beta,
    const std::array<Index, 2>& padding,
    const std::array<Index, 2>& stride,
    const std::array<Index, 2>& dilation)
{
    using Y = typename T::repr_t;
    auto dy_shape = conv2d_output_shape(x_shape, dc_shape, padding, stride, dilation);

    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());
    const Index dy_nelems = std::accumulate(
        dy_shape.begin(), dy_shape.end(), Index(1), std::multiplies<>());
    const Index dc_nelems = std::accumulate(
        dc_shape.begin(), dc_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("conv2d_bwd_weight_inplace_test");
    auto* x_node = graph.data(x_shape, "x", DataType::FP32);
    auto* dy_node = graph.data(dy_shape, "dy", DataType::FP32);
    auto* dc_node = graph.data(dc_shape, "dc", DataType::FP32);
    x_node->mark_input(true);
    dy_node->mark_input(true);
    dc_node->mark_input(true);
    dc_node->mark_output(true);

    gt::conv2d_bwd_weight_inplace(alpha, x_node, dy_node, beta, dc_node,
                             padding, stride, dilation);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> x_data(x_nelems);
    std::vector<float> dy_data(dy_nelems);
    std::vector<float> dc_data(dc_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i % 5);
    }
    for(Index i = 0; i < dy_nelems; ++i)
    {
        dy_data[i] = 0.1f * static_cast<float>(i % 3);
    }
    for(Index i = 0; i < dc_nelems; ++i)
    {
        dc_data[i] = (beta != 0.0) ? 0.01f * static_cast<float>(i) : 0.0f;
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("dy", dy_data);
    runtime.bind_data("dc", dc_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dc");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits x_traits(x_shape, x_shape);
    nntile::tensor::TensorTraits dy_traits(dy_shape, dy_shape);
    nntile::tensor::TensorTraits dc_traits(dc_shape, dc_shape);
    std::vector<int> distr(1, distr_rank_single);

    nntile::tensor::Tensor<T> x_t(x_traits, distr);
    nntile::tensor::Tensor<T> dy_t(dy_traits, distr);
    nntile::tensor::Tensor<T> dc_t(dc_traits, distr);

    auto init_tile = [](nntile::tensor::Tensor<T>& t, const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = static_cast<Y>(data[i]);
        }
        loc.release();
    };
    init_tile(x_t, x_data);
    init_tile(dy_t, dy_data);
    init_tile(dc_t, dc_data);

    nntile::tensor::conv2d_bwd_weight_inplace<T>(
        alpha, x_t, dy_t, beta, dc_t, padding, stride, dilation);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dc_nelems);
    {
        auto tile = dc_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dc_nelems; ++i)
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph conv2d_bwd_weight_inplace matches tensor API", "[graph][tensor]")
{
    const auto [x_shape, dc_shape, alpha, beta, padding, stride, dilation] = GENERATE(
        std::tuple{std::vector<Index>{4, 4, 2, 2}, std::vector<Index>{2, 2, 2, 2},
                   1.0, 0.0, std::array<Index, 2>{0, 0}, std::array<Index, 2>{1, 1},
                   std::array<Index, 2>{1, 1}},
        std::tuple{std::vector<Index>{5, 5, 2, 2}, std::vector<Index>{3, 3, 2, 2},
                   1.0, 1.0, std::array<Index, 2>{1, 1}, std::array<Index, 2>{1, 1},
                   std::array<Index, 2>{1, 1}});

    check_conv2d_bwd_weight_inplace_vs_tensor_api<nntile::fp32_t>(
        x_shape, dc_shape, alpha, beta, padding, stride, dilation);
}
