/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/rope_backward.cc
 * Test for compiled graph rope_backward operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/rope_backward.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph RopeBackward vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& sin_tensor = g.tensor({8}, "sin_tensor", DataType::FP32);
        auto& cos_tensor = g.tensor({8}, "cos_tensor", DataType::FP32);
        auto& dy = g.tensor({16, 4}, "dy", DataType::FP32);
        auto& dx = g.tensor({16, 4}, "dx", DataType::FP32);
        rope_backward(sin_tensor, cos_tensor, dy, dx);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits sin_traits({8}, {8});
        nntile::tensor::Tensor<T> sin_tensor(sin_traits);
        nntile::tensor::TensorTraits cos_traits({8}, {8});
        nntile::tensor::Tensor<T> cos_tensor(cos_traits);
        nntile::tensor::TensorTraits dy_traits({16, 4}, {16, 4});
        nntile::tensor::Tensor<T> dy(dy_traits);
        nntile::tensor::TensorTraits dx_traits({16, 4}, {16, 4});
        nntile::tensor::Tensor<T> dx(dx_traits);

        write_tensor(sin_tensor, inputs["sin_tensor"]);
        write_tensor(cos_tensor, inputs["cos_tensor"]);
        write_tensor(dy, inputs["dy"]);
        write_tensor(dx, inputs["dx"]);
        nntile::tensor::rope_backward<T>(sin_tensor, cos_tensor, dy, dx);
        outputs["dx"] = read_tensor(dx);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"sin_tensor", "cos_tensor", "dy", "dx"}, {"dx"}, context
    );
}
