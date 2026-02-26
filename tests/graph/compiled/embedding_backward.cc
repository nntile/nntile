/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/embedding_backward.cc
 * Test for compiled graph embedding_backward operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include <cstdint>

#include "nntile/tensor/embedding_backward.hh"
#include "nntile/graph/logical/embedding_backward.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph EmbeddingBackward vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& index = g.tensor({2, 3}, "index", DataType::INT64);
        auto& vocab = g.tensor({4, 3}, "vocab", DataType::FP32);
        auto& embed = g.tensor({4, 2, 3}, "embed", DataType::FP32);
        embedding_backward(index, embed, vocab, 0);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits index_traits({2, 3}, {2, 3});
        nntile::tensor::Tensor<nntile::int64_t> index_tensor(index_traits);
        nntile::tensor::TensorTraits vocab_traits({4, 3}, {4, 3});
        nntile::tensor::Tensor<T> vocab_tensor(vocab_traits);
        nntile::tensor::TensorTraits embed_traits({4, 2, 3}, {4, 2, 3});
        nntile::tensor::Tensor<T> embed_tensor(embed_traits);

        // Generate the same index data as bind_inputs (from overrides)
        // vocab shape [4,3] -> vocab_size=3, indices must be in [0, 2]
        std::vector<std::int64_t> index_data = {0, 1, 2, 0, 1, 2};

        write_tensor(index_tensor, index_data);
        write_tensor(vocab_tensor, inputs["vocab"]);
        write_tensor(embed_tensor, inputs["embed"]);

        nntile::tensor::embedding_backward<T>(index_tensor, embed_tensor, vocab_tensor, 0);
        outputs["vocab"] = read_tensor(vocab_tensor);
    };

    InputOverrides overrides;
    // vocab [4,3] -> vocab_size=3, indices must be in [0, 2]
    overrides.int64_inputs["index"] = {0, 1, 2, 0, 1, 2};

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"index", "vocab", "embed"}, {"vocab"}, context, {}, overrides
    );
}
