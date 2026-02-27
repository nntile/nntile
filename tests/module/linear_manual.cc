/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/module/linear_manual.cc
 * Tests for LinearManual (module with custom backward).
 *
 * @version 1.1.0
 * */

#include <nntile/graph.hh>
#include <nntile/module.hh>

#include <catch2/catch_test_macros.hpp>

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("LinearManual custom backward overrides autograd")
{
    NNGraph graph("test");
    module::LinearManual linear(
        graph, "linear", 4, 2, true, DataType::FP32);

    auto* input = graph.tensor({2, 4}, "input", DataType::FP32, true);

    // build_forward runs in GradMode::Guard, then wrap_with_module_op
    auto& output = linear.build_forward(*input);

    // Output has exactly one producer (module op), not gemm/add_fiber chain
    REQUIRE(output.has_producer());
    REQUIRE(output.producer()->inputs().size() >= 2);

    // Backward invokes LinearManual::build_backward
    graph.get_or_create_grad(&output, "output_grad");
    fill(Scalar(1.0), output.grad()->data());
    output.backward();

    REQUIRE(linear.weight_tensor()->has_grad());
    REQUIRE(linear.bias_tensor()->has_grad());
    REQUIRE(input->has_grad());
}
