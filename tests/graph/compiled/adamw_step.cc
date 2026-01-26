/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/adamw_step.cc
 * Test for compiled graph adamw_step operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/adamw_step.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph AdamWStep vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& grad = g.tensor({12}, "grad", DataType::FP32);
        auto& first_moment = g.tensor({12}, "first_moment", DataType::FP32);
        auto& second_moment = g.tensor({12}, "second_moment", DataType::FP32);
        auto& p = g.tensor({12}, "p", DataType::FP32);
        adamw_step(10, 0.9f, 0.999f, 1e-8f, 0.001f, 0.0f, grad, first_moment, second_moment, p);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits grad_traits({12}, {12});
        nntile::tensor::Tensor<T> grad(grad_traits);
        nntile::tensor::TensorTraits first_moment_traits({12}, {12});
        nntile::tensor::Tensor<T> first_moment(first_moment_traits);
        nntile::tensor::TensorTraits second_moment_traits({12}, {12});
        nntile::tensor::Tensor<T> second_moment(second_moment_traits);
        nntile::tensor::TensorTraits p_traits({12}, {12});
        nntile::tensor::Tensor<T> p(p_traits);

        write_tensor(grad, inputs["grad"]);
        write_tensor(first_moment, inputs["first_moment"]);
        write_tensor(second_moment, inputs["second_moment"]);
        write_tensor(p, inputs["p"]);

        nntile::tensor::adamw_step<T>(10, 0.9f, 0.999f, 1e-8f, 0.001f, 0.0f, grad, first_moment, second_moment, p);

        outputs["p"] = read_tensor(p);
        outputs["first_moment"] = read_tensor(first_moment);
        outputs["second_moment"] = read_tensor(second_moment);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"grad", "first_moment", "second_moment", "p"}, {"p", "first_moment", "second_moment"}, context
    );
}