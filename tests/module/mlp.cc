/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/mlp.cc
 * Tests for Mlp module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/modules/linear.h>
#   include <torch/nn/functional/activation.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/mlp.hh"
#include "nntile/graph/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("Mlp ForwardBuildsOutput", "[module]")
{
    NNGraph g("mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Mlp mlp(&g, "mlp", 3, 4, 5);

    auto children = mlp.named_children();
    REQUIRE(children.size() == 3);
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "fc1"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "activation"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "fc2"; }));

    auto* output = mlp.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 5}));
    REQUIRE(mlp.parameters_recursive().size() == 2);

    REQUIRE(g.num_ops() == 3);
    size_t gemm_count = 0;
    size_t gelu_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GEMM")
        {
            ++gemm_count;
        }
        if(op->op_name() == "GELU")
        {
            ++gelu_count;
        }
    }
    REQUIRE(gemm_count == 2);
    REQUIRE(gelu_count == 1);
}

TEST_CASE("Mlp BackwardCreatesGradients", "[module]")
{
    NNGraph g("mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Mlp mlp(&g, "mlp", 3, 4, 5);

    auto* output = mlp.forward(input);
    g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0), output->grad()->data());
    output->backward();

    REQUIRE(mlp.fc1().weight_tensor()->grad() != nullptr);
    REQUIRE(mlp.fc2().weight_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);

    size_t activation_backward_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GELU_BACKWARD")
        {
            ++activation_backward_count;
        }
    }
    REQUIRE(activation_backward_count == 1);
}

TEST_CASE("Mlp ActivationTypes", "[module]")
{
    NNGraph g("mlp_activations");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);

    Mlp mlp_gelu(&g, "mlp_gelu", 3, 4, 5, ActivationType::GELU);
    Mlp mlp_silu(&g, "mlp_silu", 3, 4, 5, ActivationType::SILU);
    Mlp mlp_relu(&g, "mlp_relu", 3, 4, 5, ActivationType::RELU);
    Mlp mlp_gelutanh(&g, "mlp_gelutanh", 3, 4, 5, ActivationType::GELUTANH);

    auto* out_gelu = mlp_gelu.forward(input);
    REQUIRE(out_gelu->shape() == std::vector<Index>({2, 5}));

    REQUIRE(mlp_gelu.activation().type() == ActivationType::GELU);
    REQUIRE(mlp_silu.activation().type() == ActivationType::SILU);
    REQUIRE(mlp_relu.activation().type() == ActivationType::RELU);
    REQUIRE(mlp_gelutanh.activation().type() == ActivationType::GELUTANH);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

namespace
{

torch::Tensor apply_activation_pt(torch::Tensor x, ActivationType t)
{
    switch(t)
    {
        case ActivationType::RELU:
            return torch::nn::functional::relu(x);
        case ActivationType::GELU:
            return torch::nn::functional::gelu(x);
        case ActivationType::SILU:
            return torch::nn::functional::silu(x);
        case ActivationType::GELUTANH:
            return torch::nn::functional::gelu(x,
                torch::nn::functional::GELUFuncOptions().approximate("tanh"));
        default:
            throw std::invalid_argument("Unsupported activation for test");
    }
}

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Mlp forward and backward match PyTorch", "[module][pytorch]")
{
    const auto [batch, in_dim, inter_dim, out_dim, activation, with_bias] =
        GENERATE(
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::RELU, true},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::RELU, false},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::GELU, true},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::GELU, false},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::SILU, true},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::SILU, false},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::GELUTANH, true},
            std::tuple{Index(2), Index(3), Index(4), Index(5),
                ActivationType::GELUTANH, false},
            std::tuple{Index(4), Index(8), Index(16), Index(8),
                ActivationType::RELU, true},
            std::tuple{Index(1), Index(5), Index(10), Index(3),
                ActivationType::GELU, false});

    const float grad_fill_val = 1.0f;
    const float tol = pytorch_tolerance;

    torch::manual_seed(42);
    auto fc1 = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto fc2 = torch::nn::Linear(
        torch::nn::LinearOptions(inter_dim, out_dim).bias(with_bias));

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto hidden_pt = apply_activation_pt(fc1->forward(input_pt), activation);
    auto out_pt = fc2->forward(hidden_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("mlp_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Mlp mlp(&g, "mlp", fc1, fc2, activation);
    auto* output = mlp.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(grad_fill_val), grad_output_tensor->data());
    output->backward();

    mlp.fc1().weight_tensor()->grad()->mark_output(true);
    mlp.fc2().weight_tensor()->grad()->mark_output(true);
    if(mlp.fc1().bias_tensor())
    {
        mlp.fc1().bias_tensor()->grad()->mark_output(true);
    }
    if(mlp.fc2().bias_tensor())
    {
        mlp.fc2().bias_tensor()->grad()->mark_output(true);
    }
    input->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output->name());
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, out_dim});

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < tol);

    auto grad_output = torch::full({batch, out_dim}, grad_fill_val,
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    std::vector<float> nntile_grad_w1 =
        runtime.get_output<float>(mlp.fc1().grad_name("weight"));
    std::vector<float> nntile_grad_w1_rowmajor =
        colmajor_to_rowmajor(nntile_grad_w1, {in_dim, inter_dim});
    auto pt_grad_w1 = fc1->weight.grad().accessor<float, 2>();
    for(Index i = 0; i < in_dim; ++i)
        for(Index j = 0; j < inter_dim; ++j)
            REQUIRE(std::abs(nntile_grad_w1_rowmajor[static_cast<size_t>(i * inter_dim + j)] -
                pt_grad_w1[static_cast<long>(j)][static_cast<long>(i)]) < tol);

    std::vector<float> nntile_grad_w2 =
        runtime.get_output<float>(mlp.fc2().grad_name("weight"));
    std::vector<float> nntile_grad_w2_rowmajor =
        colmajor_to_rowmajor(nntile_grad_w2, {inter_dim, out_dim});
    auto pt_grad_w2 = fc2->weight.grad().accessor<float, 2>();
    for(Index i = 0; i < inter_dim; ++i)
        for(Index j = 0; j < out_dim; ++j)
            REQUIRE(std::abs(nntile_grad_w2_rowmajor[static_cast<size_t>(i * out_dim + j)] -
                pt_grad_w2[static_cast<long>(j)][static_cast<long>(i)]) < tol);

    if(mlp.fc1().bias_tensor())
    {
        std::vector<float> nntile_grad_b1 =
            runtime.get_output<float>(mlp.fc1().grad_name("bias"));
        nntile::test::compare_float_vectors(nntile_grad_b1, fc1->bias.grad(), tol);
    }
    if(mlp.fc2().bias_tensor())
    {
        std::vector<float> nntile_grad_b2 =
            runtime.get_output<float>(mlp.fc2().grad_name("bias"));
        nntile::test::compare_float_vectors(nntile_grad_b2, fc2->bias.grad(), tol);
    }

    std::vector<float> nntile_grad_input =
        runtime.get_output<float>(input->grad()->name());
    std::vector<float> nntile_grad_input_rowmajor =
        colmajor_to_rowmajor(nntile_grad_input, {batch, in_dim});
    nntile::test::compare_float_vectors(nntile_grad_input_rowmajor, input_pt.grad(),
        tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Mlp from PyTorch forward-backward", "[module][pytorch]")
{
    const auto [batch, in_dim, inter_dim, out_dim, with_bias] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Index(5), true},
        std::tuple{Index(2), Index(3), Index(4), Index(5), false},
        std::tuple{Index(4), Index(8), Index(16), Index(8), true},
        std::tuple{Index(1), Index(5), Index(10), Index(3), false});

    torch::manual_seed(42);
    auto fc1 = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto fc2 = torch::nn::Linear(
        torch::nn::LinearOptions(inter_dim, out_dim).bias(with_bias));

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("mlp_fwd_bwd_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Mlp mlp(&g, "mlp", fc1, fc2, ActivationType::GELU);
    auto* output = mlp.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0f), grad_output_tensor->data());
    output->backward();

    mlp.fc1().weight_tensor()->grad()->mark_output(true);
    mlp.fc2().weight_tensor()->grad()->mark_output(true);
    if(mlp.fc1().bias_tensor())
    {
        mlp.fc1().bias_tensor()->grad()->mark_output(true);
    }
    if(mlp.fc2().bias_tensor())
    {
        mlp.fc2().bias_tensor()->grad()->mark_output(true);
    }
    if(input->has_grad())
    {
        input->grad()->mark_output(true);
    }

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == static_cast<size_t>(batch * out_dim));

    auto grad_w1 = runtime.get_output<float>(mlp.fc1().grad_name("weight"));
    REQUIRE(grad_w1.size() == static_cast<size_t>(in_dim * inter_dim));

    auto grad_w2 = runtime.get_output<float>(mlp.fc2().grad_name("weight"));
    REQUIRE(grad_w2.size() == static_cast<size_t>(inter_dim * out_dim));

    if(mlp.fc1().bias_tensor())
    {
        auto grad_b1 = runtime.get_output<float>(mlp.fc1().grad_name("bias"));
        REQUIRE(grad_b1.size() == static_cast<size_t>(inter_dim));
    }
    if(mlp.fc2().bias_tensor())
    {
        auto grad_b2 = runtime.get_output<float>(mlp.fc2().grad_name("bias"));
        REQUIRE(grad_b2.size() == static_cast<size_t>(out_dim));
    }
    if(input->has_grad())
    {
        auto grad_input = runtime.get_output<float>(input->grad()->name());
        REQUIRE(grad_input.size() == static_cast<size_t>(batch * in_dim));
    }
}

#endif // NNTILE_HAVE_TORCH
