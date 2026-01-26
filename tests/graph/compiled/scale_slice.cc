/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/scale_slice.cc
 * Test for compiled graph scale_slice operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/scale_slice.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ScaleSlice vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({3}, "x", DataType::FP32);  // Scaling tensor
        auto& y = g.tensor({3, 3}, "y", DataType::FP32);  // Input/output tensor
        scale_slice(x, y, 2.0f, 0);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({3}, {3});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits y_traits({3, 3}, {3, 3});
        nntile::tensor::Tensor<T> y(y_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(y, inputs["y"]);
        nntile::tensor::scale_slice<T>(2.0f, x, y, 0);
        outputs["y"] = read_tensor(y);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "y"}, {"y"}, context
    );
}