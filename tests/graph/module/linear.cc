/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/linear.cc
 * Tests for Linear module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/modules/linear.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/graph/module/linear.hh"
#include "nntile/graph/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("Linear ConstructorCreatesParameters", "[module]")
{
    NNGraph g("linear");

    Linear with_bias(&g, "linear_bias", 3, 4, true);
    REQUIRE(with_bias.weight_tensor() != nullptr);
    REQUIRE(with_bias.bias_tensor() != nullptr);
    REQUIRE(with_bias.weight_tensor()->shape() ==
        std::vector<Index>({3, 4}));
    REQUIRE(with_bias.bias_tensor()->shape() == std::vector<Index>({4}));
    REQUIRE(with_bias.weight_tensor()->name() == "linear_bias_weight");
    REQUIRE(with_bias.bias_tensor()->name() == "linear_bias_bias");
    REQUIRE(with_bias.parameters().size() == 2);

    Linear no_bias(&g, "linear_no_bias", 3, 4);
    REQUIRE(no_bias.weight_tensor() != nullptr);
    REQUIRE(no_bias.bias_tensor() == nullptr);
    REQUIRE(no_bias.parameters().size() == 1);
}

TEST_CASE("Linear ConstructorWithExistingTensors", "[module]")
{
    NNGraph g("linear");

    auto* weight = g.tensor({3, 4}, "shared_weight", DataType::FP32);
    auto* bias = g.tensor({4}, "shared_bias", DataType::FP32);

    Linear from_weight(&g, "linear_weight", weight);
    REQUIRE(from_weight.weight_tensor() == weight);
    REQUIRE(from_weight.bias_tensor() == nullptr);
    REQUIRE(from_weight.input_dim() == 3);
    REQUIRE(from_weight.output_dim() == 4);

    Linear from_weight_bias(&g, "linear_weight_bias", weight, bias);
    REQUIRE(from_weight_bias.weight_tensor() == weight);
    REQUIRE(from_weight_bias.bias_tensor() == bias);
    REQUIRE(from_weight_bias.parameters().size() == 2);
}

TEST_CASE("Linear ConstructorValidations", "[module]")
{
    NNGraph g("linear");

    auto* bad_weight = g.tensor({4}, "bad_weight", DataType::FP32);
    REQUIRE_THROWS_AS(
        Linear(&g, "linear_bad_weight", bad_weight),
        std::invalid_argument);

    auto* weight = g.tensor({3, 4}, "weight", DataType::FP32);
    auto* bad_bias = g.tensor({5}, "bad_bias", DataType::FP32);
    REQUIRE_THROWS_AS(
        Linear(&g, "linear_bad_bias", weight, bad_bias),
        std::invalid_argument);
}

TEST_CASE("Linear Callable", "[module]")
{
    NNGraph g("linear_callable");
    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Linear linear(&g, "linear", 3, 4, true);
    auto* output = linear(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 4}));
}

TEST_CASE("Linear BuildForwardWithBias", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Linear linear(&g, "linear", 3, 4, true);

    auto* output = linear.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 4}));
    REQUIRE(output->name() == "linear_output");
    REQUIRE(g.num_ops() >= 2);
    REQUIRE(g.ops()[0]->op_name() == "GEMM");
    REQUIRE(g.ops()[1]->op_name() == "ADD_FIBER");

    // Output producer is AddFiber functor (autograd, no module-level backward)
    REQUIRE(output->has_producer());
}

TEST_CASE("Linear BuildForwardValidatesInputDim", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 5}, "input", DataType::FP32);
    Linear linear(&g, "linear", 3, 4, false);

    REQUIRE_THROWS_AS(
        linear.forward(input),
        std::invalid_argument);
}

TEST_CASE("Linear BuildForwardRejectsScalarTensor", "[module]")
{
    NNGraph g("linear");

    auto* scalar = g.tensor({}, "scalar", DataType::FP32);
    Linear linear(&g, "linear", 3, 4, false);

    REQUIRE_THROWS_AS(
        linear.forward(scalar),
        std::invalid_argument);
}

TEST_CASE("Linear BackwardCreatesGradients", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", 3, 4, true);

    auto* output = linear.forward(input);
    g.get_or_create_grad(output, "output_grad");
    output->backward();

    REQUIRE(linear.weight_tensor()->grad() != nullptr);
    REQUIRE(linear.bias_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);

    REQUIRE(linear.weight_tensor()->grad()->shape() ==
        std::vector<Index>({3, 4}));
    REQUIRE(linear.bias_tensor()->grad()->shape() ==
        std::vector<Index>({4}));
    REQUIRE(input->grad()->shape() == std::vector<Index>({2, 3}));

    size_t gemm_count = 0;
    size_t sum_fiber_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GEMM")
        {
            ++gemm_count;
        }
        if(op->op_name() == "SUM_FIBER")
        {
            ++sum_fiber_count;
        }
    }
    REQUIRE(gemm_count == 3);
    REQUIRE(sum_fiber_count == 1);
}

#ifdef NNTILE_HAVE_TORCH
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear bind_weight applies data on compile", "[module]")
{
    NNGraph g("linear_bind");
    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", 3, 4, false);

    auto* output = linear.forward(input);
    input->mark_input(true);
    output->mark_output(true);

    // Bind weight before compile; data in NNTile (column-major) layout
    std::vector<float> weight_data(3 * 4);
    for(Index i = 0; i < 12; ++i)
        weight_data[i] = 0.1f * static_cast<float>(i + 1);
    linear.bind_weight(weight_data);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> input_data(2 * 3);
    for(Index i = 0; i < 6; ++i)
        input_data[i] = 1.0f;
    runtime.bind_data(input->name(), input_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == 8);
    // output = input @ weight: [2,3] @ [3,4] = [2,4]. Col-major out[b,j] at b+j*2
    // For input all 1s: out[b,j] = sum_i weight[i,j]. weight[i,j] at i+j*3
    float expected = 0;
    for(Index i = 0; i < 3; ++i)
        expected += weight_data[i];  // column 0
    REQUIRE(std::abs(out[0] - expected) < 1e-5f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear bind_bias applies data on compile", "[module]")
{
    NNGraph g("linear_bind_bias");
    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", 3, 4, true);

    auto* output = linear.forward(input);
    input->mark_input(true);
    output->mark_output(true);

    std::vector<float> weight_data(3 * 4, 0.0f);
    weight_data[0] = 1.0f;  // [0,0] = 1
    linear.bind_weight(weight_data);

    std::vector<float> bias_data(4);
    for(Index i = 0; i < 4; ++i)
        bias_data[i] = static_cast<float>(i + 1);
    linear.bind_bias(bias_data);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> input_data(2 * 3, 1.0f);
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == 8);
    // output = input @ weight + bias. Col-major: out[b,j] at index b + j*2
    // weight[0,0]=1, others 0; input all 1: gemm out[b,0]=1, out[b,j]=0 for j>0
    // + bias: out[0,0]=1+1=2, out[0,1]=0+2=2, out[0,2]=0+3=3, out[0,3]=0+4=4
    REQUIRE(std::abs(out[0] - 2.0f) < 1e-5f);
    REQUIRE(std::abs(out[2] - 2.0f) < 1e-5f);
    REQUIRE(std::abs(out[4] - 3.0f) < 1e-5f);
    REQUIRE(std::abs(out[6] - 4.0f) < 1e-5f);
}

using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear from PyTorch binds weight and bias in constructor", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index out_dim = 4;

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(in_dim, out_dim);

    NNGraph g("linear_from_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", linear_pt);  // constructor binds automatically
    auto* output = linear.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();  // bind hints applied from constructor
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output->name());
    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto out_pt = linear_pt->forward(input_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, out_dim});

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear forward matches PyTorch (no bias)", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index out_dim = 4;
    const float grad_fill_val = 1.0f;

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, out_dim).bias(false));

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto out_pt = linear_pt->forward(input_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("linear_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", linear_pt);
    auto* output = linear.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(grad_fill_val), grad_output_tensor->data());
    output->backward();

    linear.weight_tensor()->grad()->mark_output(true);
    input->grad()->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
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
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);

    auto grad_output = torch::full({batch, out_dim}, grad_fill_val,
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    std::vector<float> nntile_grad_weight =
        runtime.get_output<float>(linear.grad_name("weight"));
    std::vector<float> nntile_grad_weight_rowmajor =
        colmajor_to_rowmajor(nntile_grad_weight, {in_dim, out_dim});
    auto pt_grad_w = linear_pt->weight.grad().accessor<float, 2>();
    for(Index i = 0; i < in_dim; ++i)
        for(Index j = 0; j < out_dim; ++j)
            REQUIRE(std::abs(nntile_grad_weight_rowmajor[static_cast<size_t>(i * out_dim + j)] -
                pt_grad_w[static_cast<long>(j)][static_cast<long>(i)]) < pytorch_tolerance);

    std::vector<float> nntile_grad_input =
        runtime.get_output<float>(input->grad()->name());
    std::vector<float> nntile_grad_input_rowmajor =
        colmajor_to_rowmajor(nntile_grad_input, {batch, in_dim});
    nntile::test::compare_float_vectors(nntile_grad_input_rowmajor, input_pt.grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear forward matches PyTorch (with bias)", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index out_dim = 4;
    const float grad_fill_val = 1.0f;

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(in_dim, out_dim);

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto out_pt = linear_pt->forward(input_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("linear_bias_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", linear_pt);
    auto* output = linear.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(grad_fill_val), grad_output_tensor->data());
    output->backward();

    linear.weight_tensor()->grad()->mark_output(true);
    linear.bias_tensor()->grad()->mark_output(true);
    input->grad()->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
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
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);

    auto grad_output = torch::full({batch, out_dim}, grad_fill_val,
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    std::vector<float> nntile_grad_weight =
        runtime.get_output<float>(linear.grad_name("weight"));
    std::vector<float> nntile_grad_weight_rowmajor =
        colmajor_to_rowmajor(nntile_grad_weight, {in_dim, out_dim});
    auto pt_grad_w = linear_pt->weight.grad().accessor<float, 2>();
    for(Index i = 0; i < in_dim; ++i)
        for(Index j = 0; j < out_dim; ++j)
            REQUIRE(std::abs(nntile_grad_weight_rowmajor[static_cast<size_t>(i * out_dim + j)] -
                pt_grad_w[static_cast<long>(j)][static_cast<long>(i)]) < pytorch_tolerance);

    std::vector<float> nntile_grad_bias =
        runtime.get_output<float>(linear.grad_name("bias"));
    nntile::test::compare_float_vectors(nntile_grad_bias, linear_pt->bias.grad());

    std::vector<float> nntile_grad_input =
        runtime.get_output<float>(input->grad()->name());
    std::vector<float> nntile_grad_input_rowmajor =
        colmajor_to_rowmajor(nntile_grad_input, {batch, in_dim});
    nntile::test::compare_float_vectors(nntile_grad_input_rowmajor, input_pt.grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear from PyTorch forward-backward", "[module][pytorch]")
{
    const auto [batch, in_dim, out_dim, with_bias] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), true},
        std::tuple{Index(2), Index(3), Index(4), false},
        std::tuple{Index(4), Index(8), Index(8), true},
        std::tuple{Index(1), Index(5), Index(3), false});

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, out_dim).bias(with_bias));

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("linear_fwd_bwd_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(&g, "linear", linear_pt);
    auto* output = linear.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0f), grad_output_tensor->data());

    output->backward();

    linear.weight_tensor()->grad()->mark_output(true);
    if(linear.bias_tensor())
    {
        linear.bias_tensor()->grad()->mark_output(true);
    }
    if(input->has_grad())
    {
        input->grad()->mark_output(true);
    }

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == static_cast<size_t>(batch * out_dim));

    auto grad_weight = runtime.get_output<float>(linear.grad_name("weight"));
    REQUIRE(grad_weight.size() == static_cast<size_t>(in_dim * out_dim));

    if(linear.bias_tensor())
    {
        auto grad_bias = runtime.get_output<float>(linear.grad_name("bias"));
        REQUIRE(grad_bias.size() == static_cast<size_t>(out_dim));
    }
    if(input->has_grad())
    {
        auto grad_input = runtime.get_output<float>(input->grad()->name());
        REQUIRE(grad_input.size() == static_cast<size_t>(batch * in_dim));
    }
}

#endif // NNTILE_HAVE_TORCH
