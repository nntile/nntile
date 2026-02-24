/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/embedding.cc
 * Test for compiled graph embedding operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include <cstdint>

#include "nntile/tensor/embedding.hh"
#include "nntile/graph/logical/embedding.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Embedding vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& index = g.tensor({6}, "index", DataType::INT64);
        auto& vocab = g.tensor({4, 6}, "vocab", DataType::FP32);
        embedding(index, vocab, "embed");
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits index_traits({6}, {6});
        nntile::tensor::Tensor<nntile::int64_t> index_tensor(index_traits);
        nntile::tensor::TensorTraits vocab_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> vocab_tensor(vocab_traits);
        nntile::tensor::TensorTraits embed_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> embed_tensor(embed_traits);

        // Generate the same index data as bind_inputs does for INT64
        std::vector<std::int64_t> index_data(6);
        for(size_t i = 0; i < 6; ++i) {
            index_data[i] = static_cast<std::int64_t>(i % 4);  // Valid indices 0-3
        }

        write_tensor(index_tensor, index_data);
        write_tensor(vocab_tensor, inputs["vocab"]);

        nntile::tensor::embedding<T>(index_tensor, vocab_tensor, embed_tensor, 0);
        outputs["embed"] = read_tensor(embed_tensor);
    };

    InputOverrides overrides;
    overrides.int64_inputs["index"] = {0, 1, 2, 3, 0, 1};  // Valid indices 0-3 for vocab rows

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"index", "vocab"}, {"embed"}, context, {}, overrides
    );
}
