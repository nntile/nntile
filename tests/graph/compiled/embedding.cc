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

#include "nntile/tensor/embedding.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Embedding vs Tensor",
    "[graph][verification]")
{
    LogicalGraph g("test");
    auto& index = g.tensor({2, 3}, "index", DataType::INT64);
    auto& vocab = g.tensor({4, 10}, "vocab", DataType::FP32);
    auto& embed = embedding(index, vocab, "embed");

    auto compiled = CompiledGraph::compile(g);

    std::vector<long long> index_data = {0, 1, 2, 3, 4, 5};
    std::vector<float> vocab_data = make_pattern<float>(40, 0.05f);

    compiled.bind_data("index", index_data);
    compiled.bind_data("vocab", vocab_data);

    compiled.execute();
    compiled.wait();

    auto graph_out = compiled.get_output<float>("embed");

    using T = nntile::fp32_t;
    nntile::tensor::TensorTraits index_traits({2, 3}, {2, 3});
    nntile::tensor::Tensor<nntile::int64_t> index_tensor(index_traits);
    nntile::tensor::TensorTraits vocab_traits({4, 10}, {4, 10});
    nntile::tensor::Tensor<T> vocab_tensor(vocab_traits);
    nntile::tensor::TensorTraits embed_traits({4, 2, 3}, {4, 2, 3});
    nntile::tensor::Tensor<T> embed_tensor(embed_traits);

    write_tensor(index_tensor, index_data);
    write_tensor(vocab_tensor, vocab_data);

    nntile::tensor::embedding<T>(index_tensor, vocab_tensor, embed_tensor, 0);
    auto tensor_out = read_tensor(embed_tensor);

    REQUIRE(graph_out.size() == tensor_out.size());
    for(size_t i = 0; i < graph_out.size(); ++i)
    {
        REQUIRE(graph_out[i] == Catch::Approx(tensor_out[i]).epsilon(1e-5));
    }
}
