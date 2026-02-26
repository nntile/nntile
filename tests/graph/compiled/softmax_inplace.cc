/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/softmax_inplace.cc
 * Test for compiled graph softmax_inplace operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/softmax_inplace.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SoftmaxInplace vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& maxsumexp = g.tensor({2, 6}, "maxsumexp", DataType::FP32);
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        softmax_inplace(maxsumexp, x, 1.0f, 0);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits maxsumexp_traits({2, 6}, {2, 6});
        nntile::tensor::Tensor<T> maxsumexp(maxsumexp_traits);
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);

        write_tensor(maxsumexp, inputs["maxsumexp"]);
        write_tensor(x, inputs["x"]);
        nntile::tensor::softmax_inplace<T>(maxsumexp, 1.0f, x, 0);
        outputs["x"] = read_tensor(x);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"maxsumexp", "x"}, {"x"}, context
    );
}
