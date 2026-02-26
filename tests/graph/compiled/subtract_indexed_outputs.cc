/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/subtract_indexed_outputs.cc
 * Test for compiled graph subtract_indexed_outputs operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/subtract_indexed_outputs.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SubtractIndexedOutputs vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& labels = g.tensor({6}, "labels", DataType::INT64);
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        subtract_indexed_outputs(labels, x, 1.0f, -1);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                                std::map<std::string, std::vector<float>>& outputs,
                                const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits labels_traits({6}, {6});
        nntile::tensor::Tensor<nntile::int64_t> labels(labels_traits);
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);

        std::vector<std::int64_t> labels_data = {0, 1, 2, 0, 1, 2};
        write_tensor(labels, labels_data);
        write_tensor(x, inputs["x"]);
        nntile::tensor::subtract_indexed_outputs<T>(1.0f, labels, x, -1);
        outputs["x"] = read_tensor(x);
    };

    InputOverrides overrides;
    overrides.int64_inputs["labels"] = {0, 1, 2, 0, 1, 2};

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"labels", "x"}, {"x"}, context, {}, overrides
    );
}
