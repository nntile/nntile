/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/cross_entropy.cc
 * Test NNGraph cross_entropy autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include <torch/nn/functional/loss.h>
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph cross_entropy structure", "[graph][nn_graph]")
{
    const auto [x_shape, labels_shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{5, 7}, std::vector<Index>{7}, Index(0)},
        std::tuple{std::vector<Index>{7, 5}, std::vector<Index>{7}, Index(1)},
        std::tuple{std::vector<Index>{4, 3, 2}, std::vector<Index>{3, 2}, Index(0)});

    NNGraph g("cross_entropy_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* labels = g.tensor(labels_shape, "labels", DataType::INT64, false);
    auto* loss = cross_entropy(x, labels, "loss", axis);

    REQUIRE(loss != nullptr);
    REQUIRE(loss->has_producer());
    REQUIRE(loss->shape() == std::vector<Index>{});
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().num_ops() > 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph cross_entropy backward", "[graph][nn_graph]")
{
    std::vector<Index> x_shape{5, 7};
    std::vector<Index> labels_shape{7};

    NNGraph g("cross_entropy_backward");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* labels = g.tensor(labels_shape, "labels", DataType::INT64, false);
    auto* loss = cross_entropy(x, labels, "loss", 0);

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    fill(1.0, loss_grad);
    loss->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph cross_entropy forward and backward", "[graph][nn_graph]")
{
    const auto [x_shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{5, 7}, Index(0)},
        std::tuple{std::vector<Index>{7, 5}, Index(1)});

    std::vector<Index> labels_shape;
    labels_shape.reserve(x_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(x_shape.size()); ++i)
    {
        if(i != axis)
        {
            labels_shape.push_back(x_shape[i]);
        }
    }

    NNGraph g("cross_entropy");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* labels = g.tensor(labels_shape, "labels", DataType::INT64, false);
    auto* loss = cross_entropy(x, labels, "loss", axis);

    REQUIRE(loss != nullptr);
    REQUIRE(loss->has_producer());
    REQUIRE(loss->shape() == std::vector<Index>{});

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    fill(1.0, loss_grad);
    loss->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph cross_entropy forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    // PyTorch expects input [N, C] = [batch, nclasses], target [N]
    const Index batch_size = 7;
    const Index nclasses = 5;
    std::vector<Index> x_shape{batch_size, nclasses};
    std::vector<Index> labels_shape{batch_size};

    Index x_nelems = batch_size * nclasses;
    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i - x_nelems / 2);
    }

    std::vector<std::int64_t> labels_data(batch_size);
    for(Index i = 0; i < batch_size; ++i)
    {
        labels_data[i] = i % nclasses;
    }

    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, x_shape);

    NNGraph g("cross_entropy_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* labels = g.tensor(labels_shape, "labels", DataType::INT64, false);
    auto* loss = cross_entropy(x, labels, "loss", 1);

    x->mark_input(true);
    labels->mark_input(true);
    loss->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("labels", labels_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_loss = runtime.get_output<float>("loss");
    REQUIRE(nntile_loss.size() == 1);

    std::vector<::int64_t> shape_pt(x_shape.begin(), x_shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto labels_pt = torch::from_blob(labels_data.data(),
        {static_cast<::int64_t>(batch_size)},
        torch::TensorOptions().dtype(torch::kLong));
    auto loss_pt = torch::nn::functional::cross_entropy(
        x_pt, labels_pt, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));

    REQUIRE(std::abs(nntile_loss[0] - loss_pt.item<float>()) <
            pytorch_tolerance * (std::abs(loss_pt.item<float>()) + 1e-6f));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph cross_entropy backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const Index batch_size = 7;
    const Index nclasses = 5;
    std::vector<Index> x_shape{batch_size, nclasses};
    std::vector<Index> labels_shape{batch_size};

    Index x_nelems = batch_size * nclasses;
    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = 0.15f * static_cast<float>(i - x_nelems / 3);
    }

    std::vector<std::int64_t> labels_data(batch_size);
    for(Index i = 0; i < batch_size; ++i)
    {
        labels_data[i] = i % nclasses;
    }

    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, x_shape);

    NNGraph g("cross_entropy_bwd_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* labels = g.tensor(labels_shape, "labels", DataType::INT64, false);
    auto* loss = cross_entropy(x, labels, "loss", 1);

    x->mark_input(true);
    labels->mark_input(true);

    auto [loss_grad, _] = g.get_or_create_grad(loss, "loss_grad");
    fill(1.0, loss_grad);
    loss->backward();

    x->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("labels", labels_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x_colmajor =
        runtime.get_output<float>(x->grad()->name());
    std::vector<float> nntile_grad_x =
        colmajor_to_rowmajor(nntile_grad_x_colmajor, x_shape);

    std::vector<::int64_t> shape_pt(x_shape.begin(), x_shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), shape_pt,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto labels_pt = torch::from_blob(labels_data.data(),
        {static_cast<::int64_t>(batch_size)},
        torch::TensorOptions().dtype(torch::kLong));
    auto loss_pt = torch::nn::functional::cross_entropy(
        x_pt, labels_pt, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));
    loss_pt.backward();

    compare_float_vectors(nntile_grad_x, x_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
