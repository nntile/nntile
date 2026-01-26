/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/gelu_backward.cc
 * Test for compiled graph gelu_backward operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/gelu_backward.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluBackward vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({11, 12}, "x", DataType::FP32);
        auto& dy = g.tensor({11, 12}, "dy", DataType::FP32);
        auto& dx = g.tensor({11, 12}, "dx", DataType::FP32);
        gelu_backward(x, dy, dx);
    };

    // Create custom inputs with more realistic test data
    std::map<std::string, std::vector<float>> custom_inputs;
    size_t nelems = 11 * 12;
    custom_inputs["x"].resize(nelems);
    custom_inputs["dy"].resize(nelems);
    custom_inputs["dx"].resize(nelems);
    for(size_t i = 0; i < nelems; ++i)
    {
        custom_inputs["x"][i] = static_cast<float>(i % 10 - 5);  // Values from -5 to 4
        custom_inputs["dy"][i] = static_cast<float>(i % 7 + 1);  // Values from 1 to 7
        custom_inputs["dx"][i] = 0.0f;                           // Initialize to zero
    }

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({11, 12}, {11, 12});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits dy_traits({11, 12}, {11, 12});
        nntile::tensor::Tensor<T> dy(dy_traits);
        nntile::tensor::TensorTraits dx_traits({11, 12}, {11, 12});
        nntile::tensor::Tensor<T> dx(dx_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(dy, inputs["dy"]);
        write_tensor(dx, inputs["dx"]);

        nntile::tensor::gelu_backward<T>(x, dy, dx);
        outputs["dx"] = read_tensor(dx);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "dy", "dx"}, {"dx"}, context, custom_inputs
    );
}
