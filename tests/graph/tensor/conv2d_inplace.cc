/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/conv2d_inplace.cc
 * Test TensorGraph conv2d_inplace operation against nntile::tensor::conv2d_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <array>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/conv2d_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/conv2d_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

// Y shape from X shape (W,H,C_in,N), C shape (K_x,K_y,C_in,C_out), padding, stride, dilation
std::vector<Index> conv2d_output_shape(
    const std::vector<Index>& x_shape,
    const std::vector<Index>& c_shape,
    const std::array<Index, 2>& padding,
    const std::array<Index, 2>& stride,
    const std::array<Index, 2>& dilation)
{
    Index W_out = (x_shape[0] + 2 * padding[0] - dilation[0] * (c_shape[0] - 1) - 1)
                  / stride[0] + 1;
    Index H_out = (x_shape[1] + 2 * padding[1] - dilation[1] * (c_shape[1] - 1) - 1)
                  / stride[1] + 1;
    return {W_out, H_out, c_shape[3], x_shape[3]};
}

} // anonymous namespace

template<typename T>
void check_conv2d_inplace_vs_tensor_api(
    const std::vector<Index>& x_shape,
    const std::vector<Index>& c_shape,
    Scalar alpha,
    Scalar beta,
    const std::array<Index, 2>& padding,
    const std::array<Index, 2>& stride,
    const std::array<Index, 2>& dilation)
{
    using Y = typename T::repr_t;
    auto y_shape = conv2d_output_shape(x_shape, c_shape, padding, stride, dilation);

    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());
    const Index c_nelems = std::accumulate(
        c_shape.begin(), c_shape.end(), Index(1), std::multiplies<>());
    const Index y_nelems = std::accumulate(
        y_shape.begin(), y_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("conv2d_inplace_test");
    auto* x_node = graph.data(x_shape, "x", DataType::FP32);
    auto* c_node = graph.data(c_shape, "c", DataType::FP32);
    auto* y_node = graph.data(y_shape, "y", DataType::FP32);
    x_node->mark_input(true);
    c_node->mark_input(true);
    y_node->mark_input(true);
    y_node->mark_output(true);

    gt::conv2d_inplace(alpha, x_node, c_node, beta, y_node, padding, stride, dilation);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> x_data(x_nelems);
    std::vector<float> c_data(c_nelems);
    std::vector<float> y_data(y_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i % 5);
    }
    for(Index i = 0; i < c_nelems; ++i)
    {
        c_data[i] = 0.1f * static_cast<float>(i % 3);
    }
    for(Index i = 0; i < y_nelems; ++i)
    {
        y_data[i] = (beta != 0.0) ? 0.01f * static_cast<float>(i) : 0.0f;
    }

    runtime.bind_data("x", x_data);
    runtime.bind_data("c", c_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("y");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits x_traits(x_shape, x_shape);
    nntile::tensor::TensorTraits c_traits(c_shape, c_shape);
    nntile::tensor::TensorTraits y_traits(y_shape, y_shape);
    std::vector<int> distr(1, distr_rank_single);

    nntile::tensor::Tensor<T> x_t(x_traits, distr);
    nntile::tensor::Tensor<T> c_t(c_traits, distr);
    nntile::tensor::Tensor<T> y_t(y_traits, distr);

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
    init_tile(c_t, c_data);
    init_tile(y_t, y_data);

    nntile::tensor::conv2d_inplace<T>(alpha, x_t, c_t, beta, y_t, padding, stride, dilation);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(y_nelems);
    {
        auto tile = y_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < y_nelems; ++i)
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

TEST_CASE("TensorGraph conv2d_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* x = graph.data({4, 4, 2, 2}, "x");
    auto* c = graph.data({2, 2, 2, 2}, "c");
    auto* y = graph.data({3, 3, 2, 2}, "y");

    gt::conv2d_inplace(1.0, x, c, 0.0, y, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CONV2D_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE("TensorGraph conv2d_inplace rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* x = graph.data({4, 4, 2, 2}, "x");
    auto* c = graph.data({2, 2, 2, 2}, "c");
    auto* y = graph.data({3, 3, 2, 2}, "y");

    REQUIRE_THROWS_AS(
        gt::conv2d_inplace(1.0, nullptr, c, 0.0, y, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::conv2d_inplace(1.0, x, nullptr, 0.0, y, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph conv2d_inplace matches nntile::tensor::conv2d_inplace", "[graph][tensor]")
{
    const auto [x_shape, c_shape, alpha, beta, padding, stride, dilation] = GENERATE(
        std::tuple{std::vector<Index>{4, 4, 2, 2}, std::vector<Index>{2, 2, 2, 2},
                   1.0, 0.0, std::array<Index, 2>{0, 0}, std::array<Index, 2>{1, 1},
                   std::array<Index, 2>{1, 1}},
        std::tuple{std::vector<Index>{5, 5, 2, 2}, std::vector<Index>{3, 3, 2, 2},
                   1.0, 1.0, std::array<Index, 2>{1, 1}, std::array<Index, 2>{1, 1},
                   std::array<Index, 2>{1, 1}});

    check_conv2d_inplace_vs_tensor_api<nntile::fp32_t>(
        x_shape, c_shape, alpha, beta, padding, stride, dilation);
}
