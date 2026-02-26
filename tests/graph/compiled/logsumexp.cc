/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/logsumexp.cc
 * Test for compiled graph logsumexp operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/logsumexp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph LogSumExp vs Tensor",
    "[graph][verification]")
{
    // logsumexp takes maxsumexp-format input [2, n] (max, sumexp) and produces [n]
    // Test with pre-computed maxsumexp input (like softmax test)
    auto build_graph = [](LogicalGraph& g) {
        auto& maxsumexp_in = g.tensor({2, 6}, "maxsumexp_in", DataType::FP32);
        auto& y = g.tensor({6}, "y", DataType::FP32);
        logsumexp(maxsumexp_in, y, 0);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                                std::map<std::string, std::vector<float>>& outputs,
                                const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits maxsumexp_traits({2, 6}, {2, 6});
        nntile::tensor::Tensor<T> maxsumexp_in(maxsumexp_traits);
        nntile::tensor::TensorTraits y_traits({6}, {6});
        nntile::tensor::Tensor<T> y(y_traits);

        write_tensor(maxsumexp_in, inputs["maxsumexp_in"]);
        nntile::tensor::logsumexp<T>(maxsumexp_in, y);
        outputs["y"] = read_tensor(y);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"maxsumexp_in"}, {"y"}, context
    );
}
