/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/multiply_fiber.cc
 * Test for compiled graph multiply_fiber operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/multiply_fiber.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph MultiplyFiber vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        auto& y = g.tensor({4, 1}, "y", DataType::FP32);
        multiply_fiber(2.0f, x, y, "z", 0, 0);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits y_traits({4, 1}, {4, 1});
        nntile::tensor::Tensor<T> y(y_traits);
        nntile::tensor::TensorTraits z_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> z(z_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(y, inputs["y"]);
        nntile::tensor::multiply_fiber<T>(2.0f, x, y, z, 0);
        outputs["z"] = read_tensor(z);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "y"}, {"z"}, context
    );
}