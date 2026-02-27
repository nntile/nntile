/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/gelu.cc
 * Tests for Gelu module.
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
#include "nntile/module/gelu.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Gelu BuildForward", "[module]")
{
    NNGraph g("gelu");

    auto* input = g.tensor({2, 3, 4}, "input", DataType::FP32);
    module::Gelu gelu_mod(g, "gelu");

    auto& output = gelu_mod.build_forward(*input);
    REQUIRE(output.shape() == input->shape());
    REQUIRE(output.name() == "gelu_output");
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.ops()[0]->type() == OpType::GELU);
}

TEST_CASE("Gelu BackwardCreatesInputGrad", "[module]")
{
    NNGraph g("gelu");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    module::Gelu gelu_mod(g, "gelu");

    auto& output = gelu_mod.build_forward(*input);
    g.get_or_create_grad(&output, "output_grad");
    output.backward();

    REQUIRE(input->grad() != nullptr);
    REQUIRE(input->grad()->shape() == input->shape());

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
