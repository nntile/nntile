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

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/linear.hh"

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

TEST_CASE("Linear BuildForwardWithBias", "[module]")
{
    NNGraph g("linear");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Linear linear(g, "linear", 3, 4, true);

    auto& output = linear.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 4}));
    REQUIRE(output.name() == "linear_output");
    REQUIRE(g.num_ops() >= 2);
    REQUIRE(g.ops()[0]->type() == OpType::GEMM);
    REQUIRE(g.ops()[1]->type() == OpType::ADD_FIBER);
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
        if(op->type() == OpType::GEMM)
        {
            ++gemm_count;
        }
        if(op->type() == OpType::SUM_FIBER)
        {
            ++sum_fiber_count;
        }
    }
    REQUIRE(gemm_count == 3);
    REQUIRE(sum_fiber_count == 1);
}
