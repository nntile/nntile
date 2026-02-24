/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/sgd_step.cc
 * Test for compiled graph sgd_step operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/sgd_step.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SgdStep vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& grad = g.tensor({12}, "grad", DataType::FP32);
        auto& velocity = g.tensor({12}, "velocity", DataType::FP32);
        auto& p = g.tensor({12}, "p", DataType::FP32);
        sgd_step(10, 0.9f, 0.001f, 0.0f, 0.0f, false, grad, velocity, p);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits grad_traits({12}, {12});
        nntile::tensor::Tensor<T> grad(grad_traits);
        nntile::tensor::TensorTraits velocity_traits({12}, {12});
        nntile::tensor::Tensor<T> velocity(velocity_traits);
        nntile::tensor::TensorTraits p_traits({12}, {12});
        nntile::tensor::Tensor<T> p(p_traits);

        write_tensor(grad, inputs["grad"]);
        write_tensor(velocity, inputs["velocity"]);
        write_tensor(p, inputs["p"]);

        nntile::tensor::sgd_step<T>(10, 0.9f, 0.001f, 0.0f, 0.0f, false, grad, velocity, p);

        outputs["p"] = read_tensor(p);
        outputs["velocity"] = read_tensor(velocity);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"grad", "velocity", "p"}, {"p", "velocity"}, context
    );
}
