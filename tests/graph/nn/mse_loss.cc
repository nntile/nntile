/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/mse_loss.cc
 * Test NNGraph mse_loss autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph mse_loss structure", "[graph][nn_graph]")
{
    const auto [x_shape, scale] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 5}, Scalar(0.5)},
        std::tuple{std::vector<Index>{2, 3, 4}, Scalar(1.0 / 24.0)});

    NNGraph g("mse_loss_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* loss = mse_loss(x, "loss", scale);

    REQUIRE(loss != nullptr);
    REQUIRE(loss->has_producer());
    REQUIRE(loss->shape().empty());
    REQUIRE(loss->ndim() == 0);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph mse_loss backward", "[graph][nn_graph]")
{
    const auto [x_shape, scale] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 5}, Scalar(0.5)});

    NNGraph g("mse_loss_backward");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* loss = mse_loss(x, "loss", scale);

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    gt::fill(1.0, loss_grad->data());
    loss->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph mse_loss forward and backward", "[graph][nn_graph]")
{
    const auto [x_shape, scale, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0), Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 6}, Scalar(0.5), Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 3, 4}, Scalar(1.0 / 24.0), Scalar(-1.0)});

    NNGraph g("mse_loss");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* loss = mse_loss(x, "loss", scale);

    REQUIRE(loss != nullptr);
    REQUIRE(loss->has_producer());
    REQUIRE(loss->shape().empty());

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    gt::fill(grad_fill_val, loss_grad->data());
    loss->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph mse_loss forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [x_shape, scale] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 5}, Scalar(0.5)},
        std::tuple{std::vector<Index>{2, 3, 4}, Scalar(1.0 / 24.0)});

    Index x_nelems = 1;
    for(auto s : x_shape)
        x_nelems *= s;

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("mse_loss_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* loss = mse_loss(x, "loss", scale);

    x->mark_input(true);
    loss->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_loss = runtime.get_output<float>("loss");
    std::vector<float> x_row = colmajor_to_rowmajor(x_data, x_shape);
    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());

    auto x_pt = torch::from_blob(x_row.data(), x_shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);

    // PyTorch: mse_loss = scale * sum(x^2)
    auto loss_pt = scale * (x_pt * x_pt).sum();
    float pytorch_loss = loss_pt.item<float>();

    REQUIRE(nntile_loss.size() == 1);
    REQUIRE(std::abs(nntile_loss[0] - pytorch_loss) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph mse_loss backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [x_shape, scale, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Scalar(1.0), Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 5}, Scalar(0.5), Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 3, 4}, Scalar(1.0 / 24.0), Scalar(-1.0)});

    Index x_nelems = 1;
    for(auto s : x_shape)
        x_nelems *= s;

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.15f * static_cast<float>(i);

    NNGraph g("mse_loss_bwd_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* loss = mse_loss(x, "loss", scale);

    x->mark_input(true);

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    gt::fill(grad_fill_val, loss_grad->data());
    loss->backward();

    x->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x_colmajor =
        runtime.get_output<float>(x->grad()->name());
    std::vector<float> nntile_grad_x =
        colmajor_to_rowmajor(nntile_grad_x_colmajor, x_shape);

    std::vector<float> x_row = colmajor_to_rowmajor(x_data, x_shape);
    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());

    auto x_pt = torch::from_blob(x_row.data(), x_shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);

    // PyTorch: loss = scale * sum(x^2), grad_x = 2 * scale * x * grad_loss
    auto loss_pt = scale * (x_pt * x_pt).sum();
    auto grad_output = torch::full(
        {}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    loss_pt.backward(grad_output);

    auto grad_x_pt = x_pt.grad().contiguous();
    std::vector<float> pytorch_grad_x(grad_x_pt.data_ptr<float>(),
                                      grad_x_pt.data_ptr<float>() + x_nelems);

    REQUIRE(nntile_grad_x.size() == pytorch_grad_x.size());
    for(size_t i = 0; i < nntile_grad_x.size(); ++i)
        REQUIRE(std::abs(nntile_grad_x[i] - pytorch_grad_x[i]) < pytorch_tolerance);
}

#endif // NNTILE_HAVE_TORCH
