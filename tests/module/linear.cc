/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/linear.cc
 * Tests for Linear module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/modules/linear.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/linear.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "nntile/graph/tensor/graph.hh"
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Linear ConstructorCreatesParameters", "[module]")
{
    NNGraph g("linear");

    Linear with_bias(g, "linear_bias", 3, 4, true);
    REQUIRE(with_bias.weight_tensor() != nullptr);
    REQUIRE(with_bias.bias_tensor() != nullptr);
    REQUIRE(with_bias.weight_tensor()->shape() ==
        std::vector<Index>({3, 4}));
    REQUIRE(with_bias.bias_tensor()->shape() == std::vector<Index>({4}));
    REQUIRE(with_bias.weight_tensor()->name() == "linear_bias_weight");
    REQUIRE(with_bias.bias_tensor()->name() == "linear_bias_bias");
    REQUIRE(with_bias.parameters().size() == 2);

    Linear no_bias(g, "linear_no_bias", 3, 4);
    REQUIRE(no_bias.weight_tensor() != nullptr);
    REQUIRE(no_bias.bias_tensor() == nullptr);
    REQUIRE(no_bias.parameters().size() == 1);
}

TEST_CASE("Linear ConstructorWithExistingTensors", "[module]")
{
    NNGraph g("linear");

    auto* weight = g.tensor({3, 4}, "shared_weight", DataType::FP32);
    auto* bias = g.tensor({4}, "shared_bias", DataType::FP32);

    Linear from_weight(g, "linear_weight", *weight);
    REQUIRE(from_weight.weight_tensor() == weight);
    REQUIRE(from_weight.bias_tensor() == nullptr);
    REQUIRE(from_weight.input_dim() == 3);
    REQUIRE(from_weight.output_dim() == 4);

    Linear from_weight_bias(g, "linear_weight_bias", *weight, *bias);
    REQUIRE(from_weight_bias.weight_tensor() == weight);
    REQUIRE(from_weight_bias.bias_tensor() == bias);
    REQUIRE(from_weight_bias.parameters().size() == 2);
}

TEST_CASE("Linear ConstructorValidations", "[module]")
{
    NNGraph g("linear");

    auto* bad_weight = g.tensor({4}, "bad_weight", DataType::FP32);
    REQUIRE_THROWS_AS(
        Linear(g, "linear_bad_weight", *bad_weight),
        std::invalid_argument);

    auto* weight = g.tensor({3, 4}, "weight", DataType::FP32);
    auto* bad_bias = g.tensor({5}, "bad_bias", DataType::FP32);
    REQUIRE_THROWS_AS(
        Linear(g, "linear_bad_bias", *weight, *bad_bias),
        std::invalid_argument);
}

TEST_CASE("Linear Callable", "[module]")
{
    NNGraph g("linear_callable");
    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Linear linear(g, "linear", 3, 4, true);
    auto& output = linear(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 4}));
}

TEST_CASE("Linear BuildForwardWithBias", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Linear linear(g, "linear", 3, 4, true);

    auto& output = linear.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 4}));
    REQUIRE(output.name() == "linear_output");
    REQUIRE(g.num_ops() >= 2);
    REQUIRE(g.ops()[0]->op_name() == "GEMM");
    REQUIRE(g.ops()[1]->op_name() == "ADD_FIBER");

    // Output producer is AddFiber functor (autograd, no module-level backward)
    REQUIRE(output.has_producer());
}

TEST_CASE("Linear BuildForwardValidatesInputDim", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 5}, "input", DataType::FP32);
    Linear linear(g, "linear", 3, 4, false);

    REQUIRE_THROWS_AS(
        linear.build_forward(*input),
        std::invalid_argument);
}

TEST_CASE("Linear BuildForwardRejectsScalarTensor", "[module]")
{
    NNGraph g("linear");

    auto* scalar = g.tensor({}, "scalar", DataType::FP32);
    Linear linear(g, "linear", 3, 4, false);

    REQUIRE_THROWS_AS(
        linear.build_forward(*scalar),
        std::invalid_argument);
}

TEST_CASE("Linear BackwardCreatesGradients", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    Linear linear(g, "linear", 3, 4, true);

    auto& output = linear.build_forward(*input);
    g.get_or_create_grad(&output, "output_grad");
    output.backward();

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

using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear forward matches PyTorch (no bias)", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index out_dim = 4;

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(
        torch::nn::LinearOptions(in_dim, out_dim).bias(false));

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto out_pt = linear_pt->forward(input_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("linear_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(g, "linear", linear_pt);
    auto& output = linear.build_forward(*input);

    input->mark_input(true);
    output.mark_output(true);
    linear.weight_tensor()->mark_input(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.bind_data("linear_weight",
        Linear::weight_data_from_pytorch(linear_pt->weight));
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output.name());
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, out_dim});

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Linear forward matches PyTorch (with bias)", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index out_dim = 4;

    torch::manual_seed(42);
    auto linear_pt = torch::nn::Linear(in_dim, out_dim);

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto out_pt = linear_pt->forward(input_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("linear_bias_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    Linear linear(g, "linear", linear_pt);
    auto& output = linear.build_forward(*input);

    input->mark_input(true);
    output.mark_output(true);
    linear.weight_tensor()->mark_input(true);
    linear.bias_tensor()->mark_input(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.bind_data("linear_weight",
        Linear::weight_data_from_pytorch(linear_pt->weight));
    runtime.bind_data("linear_bias",
        Linear::bias_data_from_pytorch(linear_pt->bias));
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output.name());
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, out_dim});

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

#endif // NNTILE_HAVE_TORCH
