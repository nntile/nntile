/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/gelutanh_backward.cc
 * Test for compiled graph gelutanh_backward operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/gelutanh_backward.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GelutanhBackward vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({12}, "x", DataType::FP32);
        auto& dy = g.tensor({12}, "dy", DataType::FP32);
        auto& dx = g.tensor({12}, "dx", DataType::FP32);
        gelutanh_backward(x, dy, dx);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({12}, {12});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits dy_traits({12}, {12});
        nntile::tensor::Tensor<T> dy(dy_traits);
        nntile::tensor::TensorTraits dx_traits({12}, {12});
        nntile::tensor::Tensor<T> dx(dx_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(dy, inputs["dy"]);
        write_tensor(dx, inputs["dx"]);
        nntile::tensor::gelutanh_backward<T>(x, dy, dx);
        outputs["dx"] = read_tensor(dx);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "dy", "dx"}, {"dx"}, context
    );
}