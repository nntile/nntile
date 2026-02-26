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

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Mlp ForwardBuildsOutput", "[module]")
{
    NNGraph g("mlp");

    auto& input = g.tensor({2, 3}, "input", DataType::FP32);
    Mlp mlp(g, "mlp", 3, 4, 5);

    auto children = mlp.named_children();
    REQUIRE(children.size() == 3);
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "fc1"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "gelu"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "fc2"; }));

    auto& output = mlp.build_forward(input);
    REQUIRE(output.shape() == std::vector<Index>({2, 5}));
    REQUIRE(mlp.parameters_recursive().size() == 2);

    REQUIRE(g.num_ops() == 3);
    size_t gemm_count = 0;
    size_t gelu_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::GEMM)
        {
            ++gemm_count;
        }
        if(op->type() == OpType::GELU)
        {
            ++gelu_count;
        }
    }
    REQUIRE(gemm_count == 2);
    REQUIRE(gelu_count == 1);
}

TEST_CASE("Mlp BuildBackwardCreatesGradients", "[module]")
{
    NNGraph g("mlp");

    auto& input = g.tensor({2, 3}, "input", DataType::FP32);
    Mlp mlp(g, "mlp", 3, 4, 5);

    auto& output = mlp.build_forward(input);
    g.get_or_create_grad(output, "output_grad");
    mlp.build_backward();

    REQUIRE(mlp.fc1().weight_tensor()->grad() != nullptr);
    REQUIRE(mlp.fc2().weight_tensor()->grad() != nullptr);
    REQUIRE(input.grad() != nullptr);

    size_t gelu_backward_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::GELU_BACKWARD)
        {
            ++gelu_backward_count;
        }
    }
    REQUIRE(gelu_backward_count == 1);
}

TEST_CASE("Mlp BuildBackwardRequiresForward", "[module]")
{
    NNGraph g("mlp");

    Mlp mlp(g, "mlp", 3, 4, 5);
    REQUIRE_THROWS_AS(mlp.build_backward(), std::runtime_error);
}
