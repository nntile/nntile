/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/total_sum_accum.cc
 * Test for compiled graph total_sum_accum operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/total_sum_accum.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph TotalSumAccum vs Tensor",
    "[graph][verification]")
{
    // Tensor API: src [num_classes, batch], logsumexp [batch], labels [batch]
    auto build_graph = [](LogicalGraph& g) {
        auto& logsumexp = g.tensor({6}, "logsumexp", DataType::FP32);
        auto& src = g.tensor({4, 6}, "src", DataType::FP32);
        auto& class_labels = g.tensor({6}, "class_labels", DataType::INT64);
        auto& val = g.tensor(std::vector<Index>{}, "val", DataType::FP32);
        total_sum_accum(logsumexp, src, class_labels, val, 1.0f, -1);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& float_inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits logsumexp_traits({6}, {6});
        nntile::tensor::Tensor<T> logsumexp(logsumexp_traits);
        nntile::tensor::TensorTraits src_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> src(src_traits);
        nntile::tensor::TensorTraits class_labels_traits({6}, {6});
        nntile::tensor::Tensor<nntile::int64_t> class_labels(class_labels_traits);
        nntile::tensor::TensorTraits val_traits(std::vector<Index>{}, std::vector<Index>{});
        nntile::tensor::Tensor<nntile::fp32_t> val(val_traits);

        write_tensor(logsumexp, float_inputs["logsumexp"]);
        write_tensor(src, float_inputs["src"]);
        std::vector<std::int64_t> class_labels_data(6);
        for(size_t i = 0; i < 6; ++i) {
            class_labels_data[i] = static_cast<std::int64_t>(i % 4);
        }
        write_tensor(class_labels, class_labels_data);

        write_tensor(val, float_inputs["val"]);
        nntile::tensor::total_sum_accum<T>(1.0f, logsumexp, src, class_labels, val, -1);
        outputs["val"] = read_tensor(val);
    };

    std::map<std::string, std::vector<float>> custom_inputs = {
        {"logsumexp", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}},
        {"src", make_pattern<float>(24, 0.1f)},
        {"val", {0.0f}}
    };

    InputOverrides overrides;
    overrides.int64_inputs["class_labels"] = {0, 1, 2, 3, 0, 1};

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"logsumexp", "src", "class_labels", "val"}, {"val"}, context, custom_inputs, overrides
    );
}
