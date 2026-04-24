/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/gelu.cc
 * Test NNGraph gelu autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include "pytorch_tile_helpers.hh"
#   include <torch/nn/functional/activation.h>
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelu structure", "[graph][nn_graph]")
{
    const auto shape = GENERATE(
        std::vector<Index>{2, 3},
        std::vector<Index>{4, 5});

    NNGraph g("gelu_structure");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = gelu(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "GELU");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelu backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(-1.0)});

    NNGraph g("gelu_backward");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = gelu(x, "y");

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelu forward and backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(1.0)},
        std::tuple{std::vector<Index>{6}, Scalar(2.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Scalar(-1.0)});

    NNGraph g("gelu");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelu(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_heterogeneous_rank2_6x7;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelu forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const std::vector<Index> shape = {6, 7};
    Index nelems = 6 * 7;

    std::vector<float> x_data(nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i - nelems / 2);

    NNGraph g("gelu_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelu(x, "y");

    nn_pytorch_tile_heterogeneous_rank2_6x7(x);

    x->mark_input(true);
    y->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>("y");

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_data.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto y_pt = torch::nn::functional::gelu(x_pt);
    compare_float_vectors(nntile_out, y_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelu backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto grad_fill_val = GENERATE(Scalar(1.0), Scalar(-1.0));
    const std::vector<Index> shape = {6, 7};
    Index nelems = 6 * 7;

    std::vector<float> x_data(nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.15f * static_cast<float>(i - nelems / 3);

    NNGraph g("gelu_bwd_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelu(x, "y");

    nn_pytorch_tile_heterogeneous_rank2_6x7(x);

    x->mark_input(true);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    x->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x =
        runtime.get_output<float>(x->grad()->name());

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_data.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto y_pt = torch::nn::functional::gelu(x_pt);
    auto grad_output = torch::full(shape_pt, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
