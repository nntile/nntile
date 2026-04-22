/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/rms_norm.cc
 * Test NNGraph rms_norm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include "pytorch_tile_helpers.hh"
#   include <torch/torch.h>
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rms_norm structure", "[graph][nn_graph]")
{
    const auto [shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1)},
        std::tuple{std::vector<Index>{2, 3, 4}, Index(1)});

    std::vector<Index> gamma_shape = {shape[axis]};

    NNGraph g("rms_norm_structure");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* gamma = g.tensor(gamma_shape, "gamma", DataType::FP32);
    auto* y = rms_norm(x, gamma, "y", axis);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().num_ops() > 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rms_norm backward", "[graph][nn_graph]")
{
    const auto [shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0), Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1), Scalar(-1.0)});

    std::vector<Index> gamma_shape = {shape[axis]};

    NNGraph g("rms_norm_backward");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* gamma = g.tensor(gamma_shape, "gamma", DataType::FP32);
    auto* y = rms_norm(x, gamma, "y", axis);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(gamma->has_grad());
    REQUIRE(x->grad()->shape() == shape);
    REQUIRE(gamma->grad()->shape() == gamma_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rms_norm forward and backward", "[graph][nn_graph]")
{
    const auto [shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Index(0), Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Index(1), Scalar(1.0)},
        std::tuple{std::vector<Index>{6}, Index(0), Scalar(2.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Index(1), Scalar(-1.0)});

    std::vector<Index> gamma_shape = {shape[axis]};

    NNGraph g("rms_norm");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* gamma = g.tensor(gamma_shape, "gamma", DataType::FP32, true);
    auto* y = rms_norm(x, gamma, "y", axis);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(gamma->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
    REQUIRE(gamma->grad()->shape() == gamma_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len6;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len7;
using nntile::test::nn_pytorch_tile_heterogeneous_rank2_6x7;
using nntile::test::pytorch_tolerance;

// PyTorch rms_norm normalizes over trailing dimensions. We test with axis =
// ndim-1 to match.
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rms_norm forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{6, 7}, Index(1)},
        std::tuple{std::vector<Index>{6}, Index(0)});

    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;
    Index gamma_nelems = shape[axis];

    std::vector<float> x_data(nelems);
    std::vector<float> gamma_data(gamma_nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i - nelems / 2);
    for(Index i = 0; i < gamma_nelems; ++i)
        gamma_data[i] = 1.0f + 0.01f * static_cast<float>(i);

    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, shape);

    constexpr Scalar eps = 1e-6;

    NNGraph g("rms_norm_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* gamma = g.tensor({gamma_nelems}, "gamma", DataType::FP32, true);
    auto* y = rms_norm(x, gamma, "y", axis, eps);

    if(shape.size() == 2)
    {
        nn_pytorch_tile_heterogeneous_rank2_6x7(x);
        nn_pytorch_tile_heterogeneous_1d_len7(gamma);
    }
    else
    {
        nn_pytorch_tile_heterogeneous_1d_len6(x);
        nn_pytorch_tile_heterogeneous_1d_len6(gamma);
    }

    x->mark_input(true);
    gamma->mark_input(true);
    y->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("gamma", gamma_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("y");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, shape);

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto gamma_pt = torch::from_blob(gamma_data.data(),
                                    {static_cast<long>(gamma_nelems)},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(false);

    auto y_pt = at::rms_norm(x_pt,
        torch::IntArrayRef{static_cast<long>(shape[axis])},
        gamma_pt, static_cast<double>(eps));
    compare_float_vectors(nntile_out, y_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rms_norm backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{6, 7}, Index(1), Scalar(1.0)},
        std::tuple{std::vector<Index>{6, 7}, Index(1), Scalar(-1.0)},
        std::tuple{std::vector<Index>{6}, Index(0), Scalar(1.0)});

    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;
    Index gamma_nelems = shape[axis];

    std::vector<float> x_data(nelems);
    std::vector<float> gamma_data(gamma_nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.15f * static_cast<float>(i - nelems / 3);
    for(Index i = 0; i < gamma_nelems; ++i)
        gamma_data[i] = 1.0f + 0.02f * static_cast<float>(i);

    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, shape);

    constexpr Scalar eps = 1e-6;

    NNGraph g("rms_norm_bwd_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* gamma = g.tensor({gamma_nelems}, "gamma", DataType::FP32, true);
    auto* y = rms_norm(x, gamma, "y", axis, eps);

    if(shape.size() == 2)
    {
        nn_pytorch_tile_heterogeneous_rank2_6x7(x);
        nn_pytorch_tile_heterogeneous_1d_len7(gamma);
    }
    else
    {
        nn_pytorch_tile_heterogeneous_1d_len6(x);
        nn_pytorch_tile_heterogeneous_1d_len6(gamma);
    }

    x->mark_input(true);
    gamma->mark_input(true);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    x->grad()->mark_output(true);
    gamma->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("gamma", gamma_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x_colmajor =
        runtime.get_output<float>(x->grad()->name());
    std::vector<float> nntile_grad_x =
        colmajor_to_rowmajor(nntile_grad_x_colmajor, shape);
    std::vector<float> nntile_grad_gamma =
        runtime.get_output<float>(gamma->grad()->name());

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto gamma_pt = torch::from_blob(gamma_data.data(),
                                     {static_cast<long>(gamma_nelems)},
                                     torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(true);

    auto y_pt = at::rms_norm(x_pt,
        torch::IntArrayRef{static_cast<long>(shape[axis])},
        gamma_pt, static_cast<double>(eps));
    auto grad_output = torch::full(shape_pt, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
    compare_float_vectors(nntile_grad_gamma, gamma_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
