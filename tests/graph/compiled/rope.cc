/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/rope.cc
 * Test for compiled graph rope operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/rope.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Rope vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& sin_tensor = g.tensor({8}, "sin_tensor", DataType::FP32);
        auto& cos_tensor = g.tensor({8}, "cos_tensor", DataType::FP32);
        auto& src = g.tensor({16, 4}, "src", DataType::FP32);
        auto& dst = g.tensor({16, 4}, "dst", DataType::FP32);
        rope(sin_tensor, cos_tensor, src, dst);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits sin_traits({8}, {8});
        nntile::tensor::Tensor<T> sin_tensor(sin_traits);
        nntile::tensor::TensorTraits cos_traits({8}, {8});
        nntile::tensor::Tensor<T> cos_tensor(cos_traits);
        nntile::tensor::TensorTraits src_traits({16, 4}, {16, 4});
        nntile::tensor::Tensor<T> src(src_traits);
        nntile::tensor::TensorTraits dst_traits({16, 4}, {16, 4});
        nntile::tensor::Tensor<T> dst(dst_traits);

        write_tensor(sin_tensor, inputs["sin_tensor"]);
        write_tensor(cos_tensor, inputs["cos_tensor"]);
        write_tensor(src, inputs["src"]);
        nntile::tensor::rope<T>(sin_tensor, cos_tensor, src, dst);
        outputs["dst"] = read_tensor(dst);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"sin_tensor", "cos_tensor", "src"}, {"dst"}, context
    );
}
