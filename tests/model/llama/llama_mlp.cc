/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/llama/llama_mlp.cc
 * Tests for LlamaMLP module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/model/llama/llama_mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
namespace gt = nntile::graph::tensor;

TEST_CASE("LlamaMLP ForwardBuildsOutput", "[model][llama]")
{
    NNGraph g("llama_mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    LlamaMLP mlp(g, "llama_mlp", 3, 4, 5);

    auto children = mlp.named_children();
    REQUIRE(children.size() == 3);
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "gate_proj"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "up_proj"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "down_proj"; }));

    auto& output = mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 5}));
    REQUIRE(mlp.parameters_recursive().size() == 3);

    size_t gemm_count = 0;
    size_t silu_count = 0;
    size_t multiply_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GEMM")
        {
            ++gemm_count;
        }
        if(op->op_name() == "SILU")
        {
            ++silu_count;
        }
        if(op->op_name() == "MULTIPLY")
        {
            ++multiply_count;
        }
    }
    REQUIRE(gemm_count == 3);
    REQUIRE(silu_count == 1);
    REQUIRE(multiply_count == 1);
}

TEST_CASE("LlamaMLP BackwardCreatesGradients", "[model][llama]")
{
    NNGraph g("llama_mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    LlamaMLP mlp(g, "llama_mlp", 3, 4, 5);

    auto& output = mlp.build_forward(*input);
    g.get_or_create_grad(&output, "output_grad");
    gt::fill(Scalar(1.0), output.grad()->data());
    output.backward();

    REQUIRE(mlp.gate_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(mlp.up_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(mlp.down_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);

    size_t silu_backward_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "SILU_BACKWARD")
        {
            ++silu_backward_count;
        }
    }
    REQUIRE(silu_backward_count == 1);
}

TEST_CASE("LlamaMLP SquareDimensions", "[model][llama]")
{
    NNGraph g("llama_mlp");

    auto* input = g.tensor({2, 128}, "input", DataType::FP32);
    LlamaMLP mlp(g, "llama_mlp", 128, 256);

    auto& output = mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 128}));
    REQUIRE(mlp.input_dim() == 128);
    REQUIRE(mlp.intermediate_dim() == 256);
    REQUIRE(mlp.output_dim() == 128);
}
