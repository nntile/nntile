/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/gelutanh.cc
 * Test NNGraph gelutanh autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include <torch/nn/functional/activation.h>
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelutanh structure", "[graph][nn_graph]")
{
    const auto shape = GENERATE(
        std::vector<Index>{2, 3},
        std::vector<Index>{4, 5});

    NNGraph g("gelutanh_structure");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = gelutanh(x, "y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "GELUTANH");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelutanh backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(-1.0)});

    NNGraph g("gelutanh_backward");
    auto* x = g.tensor(shape, "x", DataType::FP32);
    auto* y = gelutanh(x, "y");

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelutanh forward and backward", "[graph][nn_graph]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 5}, Scalar(1.0)},
        std::tuple{std::vector<Index>{6}, Scalar(2.0)},
        std::tuple{std::vector<Index>{2, 2, 3}, Scalar(-1.0)});

    NNGraph g("gelutanh");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelutanh(x, "y");

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
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelutanh forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 6},
        std::vector<Index>{2, 3, 4});

    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;

    std::vector<float> x_data(nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i - nelems / 2);

    NNGraph g("gelutanh_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelutanh(x, "y");

    x->mark_input(true);
    y->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
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
    auto y_pt = torch::nn::functional::gelu(x_pt,
        torch::nn::functional::GELUFuncOptions().approximate("tanh"));
    std::vector<float> pytorch_out(y_pt.data_ptr<float>(),
                                   y_pt.data_ptr<float>() + nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gelutanh backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [shape, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{3, 5}, Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 4}, Scalar(-1.0)});

    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;

    std::vector<float> x_data(nelems);
    for(Index i = 0; i < nelems; ++i)
        x_data[i] = 0.15f * static_cast<float>(i - nelems / 3);

    NNGraph g("gelutanh_bwd_pytorch");
    auto* x = g.tensor(shape, "x", DataType::FP32, true);
    auto* y = gelutanh(x, "y");

    x->mark_input(true);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    x->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
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
    auto y_pt = torch::nn::functional::gelu(x_pt,
        torch::nn::functional::GELUFuncOptions().approximate("tanh"));
    auto grad_output = torch::full(shape_pt, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
