/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/gpt2/gpt2_mlp.cc
 * Tests for Gpt2Mlp module.
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
#include "nntile/model/gpt2/gpt2_mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::gpt2;
namespace gt = nntile::graph::tensor;

TEST_CASE("Gpt2Mlp ForwardBuildsOutput", "[model][gpt2]")
{
    NNGraph g("gpt2_mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Gpt2Mlp gpt2_mlp(g, "gpt2_mlp", 3, 12, 3);  // GPT-2 style: 4x expansion

    auto children = gpt2_mlp.named_children();
    REQUIRE(children.size() == 3);
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "c_fc"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "gelu"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "c_proj"; }));

    auto& output = gpt2_mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 3}));
    REQUIRE(gpt2_mlp.parameters_recursive().size() == 2);

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

TEST_CASE("Gpt2Mlp BackwardCreatesGradients", "[model][gpt2]")
{
    NNGraph g("gpt2_mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    Gpt2Mlp gpt2_mlp(g, "gpt2_mlp", 3, 12, 3);

    auto& output = gpt2_mlp.build_forward(*input);
    g.get_or_create_grad(&output, "output_grad");
    gt::fill(Scalar(1.0), output.grad()->data());
    output.backward();

    REQUIRE(gpt2_mlp.c_fc().weight_tensor()->grad() != nullptr);
    REQUIRE(gpt2_mlp.c_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);

    size_t gelu_backward_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GELU_BACKWARD")
        {
            ++gelu_backward_count;
        }
    }
    REQUIRE(gelu_backward_count == 1);
}

TEST_CASE("Gpt2Mlp Repr", "[model][gpt2]")
{
    NNGraph g("gpt2_mlp");
    Gpt2Mlp gpt2_mlp(g, "gpt2_mlp", 768, 3072, 768);

    std::string r = gpt2_mlp.repr();
    REQUIRE(r.find("Gpt2Mlp") != std::string::npos);
    REQUIRE(r.find("768") != std::string::npos);
    REQUIRE(r.find("3072") != std::string::npos);
}

TEST_CASE("Gpt2Mlp ConstructorOutputEqualsInput", "[model][gpt2]")
{
    NNGraph g("gpt2_mlp");

    auto* input = g.tensor({4, 256}, "input", DataType::FP32);
    // Use constructor where output_dim == input_dim
    Gpt2Mlp gpt2_mlp(g, "gpt2_mlp", 256, 1024);  // 4x expansion, output=256

    auto& output = gpt2_mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({4, 256}));
    REQUIRE(gpt2_mlp.input_dim() == 256);
    REQUIRE(gpt2_mlp.intermediate_dim() == 1024);
    REQUIRE(gpt2_mlp.output_dim() == 256);
}
