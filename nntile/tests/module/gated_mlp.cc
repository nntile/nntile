/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/module/gated_mlp.cc
 * Tests for GatedMlp module.
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
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/linear.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/gated_mlp.hh"
#include "nntile/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#include "context_fixture.hh"
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

using namespace nntile;
using namespace nntile;
using namespace nntile::module;
namespace gt = nntile::tensor;

TEST_CASE("GatedMlp ForwardBuildsOutput", "[module]")
{
    NNGraph g("gated_mlp");

    auto *input = g.tensor({2, 3}, DataType::FP32)->set_name("input");
    GatedMlp gated_mlp(&g, "gated_mlp", 3, 4, 5);

    auto children = gated_mlp.named_children();
    REQUIRE(children.size() == 4);
    REQUIRE(std::any_of(children.begin(),
        children.end(),
        [](const auto &entry) { return entry.first == "gate_proj"; }));
    REQUIRE(std::any_of(children.begin(),
        children.end(),
        [](const auto &entry) { return entry.first == "up_proj"; }));
    REQUIRE(std::any_of(children.begin(),
        children.end(),
        [](const auto &entry) { return entry.first == "activation"; }));
    REQUIRE(std::any_of(children.begin(),
        children.end(),
        [](const auto &entry) { return entry.first == "down_proj"; }));

    auto *output = gated_mlp.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 5}));
    REQUIRE(gated_mlp.parameters_recursive().size() == 3);

    REQUIRE(gated_mlp.activation().type() == ActivationType::SILU);

    size_t gemm_count = 0;
    size_t silu_count = 0;
    size_t multiply_count = 0;
    for (const auto &op : g.ops())
    {
        if (op->op_name() == "GEMM")
        {
            ++gemm_count;
        }
        if (op->op_name() == "SILU")
        {
            ++silu_count;
        }
        if (op->op_name() == "MULTIPLY")
        {
            ++multiply_count;
        }
    }
    REQUIRE(gemm_count == 3);
    REQUIRE(silu_count == 1);
    REQUIRE(multiply_count == 1);
}

TEST_CASE("GatedMlp BackwardCreatesGradients", "[module]")
{
    NNGraph g("gated_mlp_bwd");

    auto *input = g.tensor({2, 3}, DataType::FP32)->set_name("input");
    GatedMlp gated_mlp(&g, "gated_mlp", 3, 4, 5);

    auto *output = gated_mlp.forward(input);
    g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0), output->grad()->data());
    output->backward();

    REQUIRE(gated_mlp.gate_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(gated_mlp.up_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(gated_mlp.down_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);
}

TEST_CASE("GatedMlp OutputDimEqualsInputDim", "[module]")
{
    NNGraph g("gated_mlp_square");

    auto *input = g.tensor({2, 8}, DataType::FP32)->set_name("input");
    GatedMlp gated_mlp(&g, "gated_mlp", 8, 16);

    auto *output = gated_mlp.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 8}));
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

namespace
{

torch::Tensor apply_activation_pt(torch::Tensor x, ActivationType t)
{
    switch (t)
    {
    case ActivationType::RELU:
        return torch::nn::functional::relu(x);
    case ActivationType::GELU:
        return torch::nn::functional::gelu(x);
    case ActivationType::SILU:
        return torch::nn::functional::silu(x);
    case ActivationType::GELUTANH:
        return torch::nn::functional::gelu(
            x, torch::nn::functional::GELUFuncOptions().approximate("tanh"));
    default:
        throw std::invalid_argument("Unsupported activation for test");
    }
}

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GatedMlp forward and backward match PyTorch",
    "[module][pytorch]")
{
    const auto [batch, in_dim, inter_dim, out_dim, activation, with_bias] =
        GENERATE(std::tuple{Index(2),
                     Index(3),
                     Index(4),
                     Index(5),
                     ActivationType::SILU,
                     true},
            std::tuple{Index(2),
                Index(3),
                Index(4),
                Index(5),
                ActivationType::SILU,
                false},
            std::tuple{Index(2),
                Index(3),
                Index(4),
                Index(5),
                ActivationType::RELU,
                true},
            std::tuple{Index(2),
                Index(3),
                Index(4),
                Index(5),
                ActivationType::GELU,
                true},
            std::tuple{Index(4),
                Index(8),
                Index(16),
                Index(8),
                ActivationType::SILU,
                true},
            std::tuple{Index(1),
                Index(5),
                Index(10),
                Index(3),
                ActivationType::SILU,
                false});

    const float grad_fill_val = 1.0f;
    const float tol = pytorch_tolerance;

    torch::manual_seed(42);
    auto gate_proj = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto up_proj = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto down_proj = torch::nn::Linear(
        torch::nn::LinearOptions(inter_dim, out_dim).bias(with_bias));

    std::vector<float> input_data(batch * in_dim);
    for (Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    auto input_pt = torch::from_blob(input_data.data(),
        {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(true);
    auto gate_pt = gate_proj->forward(input_pt);
    auto up_pt = up_proj->forward(input_pt);
    auto hidden_pt = apply_activation_pt(gate_pt, activation) * up_pt;
    auto out_pt = down_proj->forward(hidden_pt);

    NNGraph g("gated_mlp_pytorch");
    auto *input =
        g.tensor({batch, in_dim}, DataType::FP32, true)->set_name("input");
    GatedMlp gated_mlp(
        &g, "gated_mlp", gate_proj, up_proj, down_proj, activation);
    auto *output = gated_mlp.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(grad_fill_val), grad_output_tensor->data());
    output->backward();

    gated_mlp.gate_proj().weight_tensor()->grad()->mark_output(true);
    gated_mlp.up_proj().weight_tensor()->grad()->mark_output(true);
    gated_mlp.down_proj().weight_tensor()->grad()->mark_output(true);
    if (gated_mlp.gate_proj().bias_tensor())
    {
        gated_mlp.gate_proj().bias_tensor()->grad()->mark_output(true);
    }
    if (gated_mlp.up_proj().bias_tensor())
    {
        gated_mlp.up_proj().bias_tensor()->grad()->mark_output(true);
    }
    if (gated_mlp.down_proj().bias_tensor())
    {
        gated_mlp.down_proj().bias_tensor()->grad()->mark_output(true);
    }
    input->grad()->mark_output(true);

    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());

    Runtime runtime(tile_graph);
    runtime.compile();
    g.bind_parameters(runtime);
    runtime.bind_data(input, input_data);
    runtime.execute();
    runtime.wait();

    compare_float_vectors(runtime.get_output<float>(output), out_pt, tol);

    auto grad_output = torch::full({batch, out_dim},
        grad_fill_val,
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(
        runtime.get_output<float>(
            gated_mlp.gate_proj().weight_tensor()->grad()),
        gate_proj->weight.grad(),
        tol);
    compare_float_vectors(
        runtime.get_output<float>(gated_mlp.up_proj().weight_tensor()->grad()),
        up_proj->weight.grad(),
        tol);
    compare_float_vectors(
        runtime.get_output<float>(
            gated_mlp.down_proj().weight_tensor()->grad()),
        down_proj->weight.grad(),
        tol);

    if (gated_mlp.gate_proj().bias_tensor())
    {
        std::vector<float> nntile_grad_b = runtime.get_output<float>(
            gated_mlp.gate_proj().bias_tensor()->grad());
        nntile::test::compare_float_vectors(
            nntile_grad_b, gate_proj->bias.grad(), tol);
    }
    if (gated_mlp.up_proj().bias_tensor())
    {
        std::vector<float> nntile_grad_b = runtime.get_output<float>(
            gated_mlp.up_proj().bias_tensor()->grad());
        nntile::test::compare_float_vectors(
            nntile_grad_b, up_proj->bias.grad(), tol);
    }
    if (gated_mlp.down_proj().bias_tensor())
    {
        std::vector<float> nntile_grad_b = runtime.get_output<float>(
            gated_mlp.down_proj().bias_tensor()->grad());
        nntile::test::compare_float_vectors(
            nntile_grad_b, down_proj->bias.grad(), tol);
    }

    compare_float_vectors(
        runtime.get_output<float>(input->grad()), input_pt.grad(), tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GatedMlp from PyTorch forward-backward",
    "[module][pytorch]")
{
    const auto [batch, in_dim, inter_dim, out_dim, with_bias] =
        GENERATE(std::tuple{Index(2), Index(3), Index(4), Index(5), true},
            std::tuple{Index(2), Index(3), Index(4), Index(5), false},
            std::tuple{Index(4), Index(8), Index(16), Index(8), true},
            std::tuple{Index(1), Index(5), Index(10), Index(3), false});

    torch::manual_seed(42);
    auto gate_proj = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto up_proj = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, inter_dim).bias(with_bias));
    auto down_proj = torch::nn::Linear(
        torch::nn::LinearOptions(inter_dim, out_dim).bias(with_bias));

    std::vector<float> input_data(batch * in_dim);
    for (Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("gated_mlp_fwd_bwd_pytorch");
    auto *input =
        g.tensor({batch, in_dim}, DataType::FP32, true)->set_name("input");
    GatedMlp gated_mlp(
        &g, "gated_mlp", gate_proj, up_proj, down_proj, ActivationType::SILU);
    auto *output = gated_mlp.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0f), grad_output_tensor->data());
    output->backward();

    gated_mlp.gate_proj().weight_tensor()->grad()->mark_output(true);
    gated_mlp.up_proj().weight_tensor()->grad()->mark_output(true);
    gated_mlp.down_proj().weight_tensor()->grad()->mark_output(true);
    if (gated_mlp.gate_proj().bias_tensor())
    {
        gated_mlp.gate_proj().bias_tensor()->grad()->mark_output(true);
    }
    if (gated_mlp.up_proj().bias_tensor())
    {
        gated_mlp.up_proj().bias_tensor()->grad()->mark_output(true);
    }
    if (gated_mlp.down_proj().bias_tensor())
    {
        gated_mlp.down_proj().bias_tensor()->grad()->mark_output(true);
    }
    if (input->has_grad())
    {
        input->grad()->mark_output(true);
    }

    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());

    Runtime runtime(tile_graph);
    runtime.compile();
    g.bind_parameters(runtime);
    runtime.bind_data(input, input_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output);
    REQUIRE(out.size() == static_cast<size_t>(batch * out_dim));

    auto grad_gate = runtime.get_output<float>(
        gated_mlp.gate_proj().weight_tensor()->grad());
    REQUIRE(grad_gate.size() == static_cast<size_t>(inter_dim * in_dim));

    auto grad_up =
        runtime.get_output<float>(gated_mlp.up_proj().weight_tensor()->grad());
    REQUIRE(grad_up.size() == static_cast<size_t>(inter_dim * in_dim));

    auto grad_down = runtime.get_output<float>(
        gated_mlp.down_proj().weight_tensor()->grad());
    REQUIRE(grad_down.size() == static_cast<size_t>(out_dim * inter_dim));

    if (gated_mlp.gate_proj().bias_tensor())
    {
        auto grad_b = runtime.get_output<float>(
            gated_mlp.gate_proj().bias_tensor()->grad());
        REQUIRE(grad_b.size() == static_cast<size_t>(inter_dim));
    }
    if (gated_mlp.up_proj().bias_tensor())
    {
        auto grad_b = runtime.get_output<float>(
            gated_mlp.up_proj().bias_tensor()->grad());
        REQUIRE(grad_b.size() == static_cast<size_t>(inter_dim));
    }
    if (gated_mlp.down_proj().bias_tensor())
    {
        auto grad_b = runtime.get_output<float>(
            gated_mlp.down_proj().bias_tensor()->grad());
        REQUIRE(grad_b.size() == static_cast<size_t>(out_dim));
    }
    if (input->has_grad())
    {
        auto grad_input = runtime.get_output<float>(input->grad());
        REQUIRE(grad_input.size() == static_cast<size_t>(batch * in_dim));
    }
}

#endif // NNTILE_HAVE_TORCH
