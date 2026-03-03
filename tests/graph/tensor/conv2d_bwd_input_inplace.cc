/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/conv2d_bwd_input_inplace.cc
 * Test TensorGraph conv2d_bwd_input_inplace operation against
 * nntile::tensor::conv2d_bwd_input_inplace.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <array>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/conv2d_bwd_input_inplace.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/conv2d_bwd_input_inplace.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

// dY shape (W_out,H_out,C_out,N) from X shape, C shape, padding, stride, dilation
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
void check_conv2d_bwd_input_inplace_vs_tensor_api(
    const std::vector<Index>& dx_shape,
    const std::vector<Index>& kernel_shape,
    Scalar alpha,
    Scalar beta,
    const std::array<Index, 2>& padding,
    const std::array<Index, 2>& stride,
    const std::array<Index, 2>& dilation)
{
    using Y = typename T::repr_t;
    auto dy_shape = conv2d_output_shape(dx_shape, kernel_shape, padding, stride, dilation);

    const Index dy_nelems = std::accumulate(
        dy_shape.begin(), dy_shape.end(), Index(1), std::multiplies<>());
    const Index kernel_nelems = std::accumulate(
        kernel_shape.begin(), kernel_shape.end(), Index(1), std::multiplies<>());
    const Index dx_nelems = std::accumulate(
        dx_shape.begin(), dx_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("conv2d_bwd_input_test");
    auto* dy_node = graph.data(dy_shape, "dy", DataType::FP32);
    auto* kernel_node = graph.data(kernel_shape, "kernel", DataType::FP32);
    auto* dx_node = graph.data(dx_shape, "dx", DataType::FP32);
    dy_node->mark_input(true);
    kernel_node->mark_input(true);
    dx_node->mark_input(true);
    dx_node->mark_output(true);

    conv2d_bwd_input_inplace(alpha, dy_node, kernel_node, beta, dx_node,
                             padding, stride, dilation);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> dy_data(dy_nelems);
    std::vector<float> kernel_data(kernel_nelems);
    std::vector<float> dx_data(dx_nelems);
    for(Index i = 0; i < dy_nelems; ++i)
    {
        dy_data[i] = 0.1f * static_cast<float>(i % 5);
    }
    for(Index i = 0; i < kernel_nelems; ++i)
    {
        kernel_data[i] = 0.1f * static_cast<float>(i % 3);
    }
    for(Index i = 0; i < dx_nelems; ++i)
    {
        dx_data[i] = (beta != 0.0) ? 0.01f * static_cast<float>(i) : 0.0f;
    }

    runtime.bind_data("dy", dy_data);
    runtime.bind_data("kernel", kernel_data);
    runtime.bind_data("dx", dx_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dx");

    // --- Direct tensor API path ---
    tensor::TensorTraits dy_traits(dy_shape, dy_shape);
    tensor::TensorTraits kernel_traits(kernel_shape, kernel_shape);
    tensor::TensorTraits dx_traits(dx_shape, dx_shape);
    std::vector<int> distr(1, distr_rank_single);

    tensor::Tensor<T> dy_t(dy_traits, distr);
    tensor::Tensor<T> kernel_t(kernel_traits, distr);
    tensor::Tensor<T> dx_t(dx_traits, distr);

    auto init_tile = [](tensor::Tensor<T>& t, const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = static_cast<Y>(data[i]);
        }
        loc.release();
    };
    init_tile(dy_t, dy_data);
    init_tile(kernel_t, kernel_data);
    init_tile(dx_t, dx_data);

    tensor::conv2d_bwd_input_inplace<T>(alpha, dy_t, kernel_t, beta, dx_t,
                                        padding, stride, dilation);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dx_nelems);
    {
        auto tile = dx_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dx_nelems; ++i)
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

TEST_CASE("TensorGraph conv2d_bwd_input_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* dy = graph.data({3, 3, 2, 2}, "dy");
    auto* kernel = graph.data({2, 2, 2, 2}, "kernel");
    auto* dx = graph.data({4, 4, 2, 2}, "dx");

    conv2d_bwd_input_inplace(1.0, dy, kernel, 0.0, dx, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CONV2D_BWD_INPUT_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dx);
}

TEST_CASE("TensorGraph conv2d_bwd_input_inplace rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* dy = graph.data({3, 3, 2, 2}, "dy");
    auto* kernel = graph.data({2, 2, 2, 2}, "kernel");
    auto* dx = graph.data({4, 4, 2, 2}, "dx");

    REQUIRE_THROWS_AS(
        conv2d_bwd_input_inplace(1.0, nullptr, kernel, 0.0, dx, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        conv2d_bwd_input_inplace(1.0, dy, nullptr, 0.0, dx, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph conv2d_bwd_input_inplace matches tensor::conv2d_bwd_input_inplace",
    "[graph][tensor]")
{
    const auto [dx_shape, kernel_shape, alpha, beta, padding, stride, dilation] =
        GENERATE(
            std::tuple{std::vector<Index>{4, 4, 2, 2}, std::vector<Index>{2, 2, 2, 2},
                       1.0, 0.0, std::array<Index, 2>{0, 0}, std::array<Index, 2>{1, 1},
                       std::array<Index, 2>{1, 1}},
            std::tuple{std::vector<Index>{5, 5, 2, 2}, std::vector<Index>{3, 3, 2, 2},
                       1.0, 1.0, std::array<Index, 2>{1, 1}, std::array<Index, 2>{1, 1},
                       std::array<Index, 2>{1, 1}});

    check_conv2d_bwd_input_inplace_vs_tensor_api<nntile::fp32_t>(
        dx_shape, kernel_shape, alpha, beta, padding, stride, dilation);
}
