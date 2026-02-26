/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/norm_fiber.cc
 * Test for compiled graph norm_fiber operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/norm_fiber.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph NormFiber vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        auto& src2 = g.tensor({4}, "src2", DataType::FP32);
        auto& dst = g.tensor({4}, "dst", DataType::FP32);
        norm_fiber(x, src2, dst, 0, 0, 0, 2.0f, 3.0f);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits src2_traits({4}, {4});
        nntile::tensor::Tensor<T> src2(src2_traits);
        nntile::tensor::TensorTraits dst_traits({4}, {4});
        nntile::tensor::Tensor<T> dst(dst_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(src2, inputs["src2"]);
        nntile::tensor::norm_fiber<T>(2.0f, x, 3.0f, src2, dst, 0, 0);
        outputs["dst"] = read_tensor(dst);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "src2"}, {"dst"}, context
    );
}
