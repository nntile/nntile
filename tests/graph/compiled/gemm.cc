/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/gemm.cc
 * Test for compiled graph gemm operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/constants.hh"
#include "nntile/tensor/gemm.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GemmAccumulation vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& a = g.tensor({2, 3}, "a", DataType::FP32);
        auto& b = g.tensor({3, 4}, "b", DataType::FP32);
        auto& c = g.tensor({2, 4}, "c", DataType::FP32);
        gemm(a, b, c, 2.0f, 0.5f);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits a_traits({2, 3}, {2, 3});
        nntile::tensor::Tensor<T> a(a_traits);
        nntile::tensor::TensorTraits b_traits({3, 4}, {3, 4});
        nntile::tensor::Tensor<T> b(b_traits);
        nntile::tensor::TensorTraits c_traits({2, 4}, {2, 4});
        nntile::tensor::Tensor<T> c(c_traits);

        write_tensor(a, inputs["a"]);
        write_tensor(b, inputs["b"]);
        write_tensor(c, inputs["c"]);

        nntile::TransOp trans_a = nntile::TransOp(nntile::TransOp::NoTrans);
        nntile::TransOp trans_b = nntile::TransOp(nntile::TransOp::NoTrans);
        nntile::tensor::gemm<T>(2.0f, trans_a, a, trans_b, b, 0.5f, c, 1, 0, 0);

        outputs["c"] = read_tensor(c);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"a", "b", "c"}, {"c"}, context
    );
}
