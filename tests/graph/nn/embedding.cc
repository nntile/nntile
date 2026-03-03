/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/embedding.cc
 * Test NNGraph embedding autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

// embed shape: index.shape + vocab.dim(1), axis = index.ndim
static std::vector<Index> embed_output_shape(
    const std::vector<Index>& index_shape,
    const std::vector<Index>& vocab_shape)
{
    std::vector<Index> embed_shape = index_shape;
    embed_shape.push_back(vocab_shape[1]);
    return embed_shape;
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding structure", "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 100},
                   Index(2)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 50}, Index(1)});

    NNGraph g("embedding_structure");
    auto* index = g.tensor(index_shape, "index", DataType::INT64, false);
    auto* vocab = g.tensor(vocab_shape, "vocab", DataType::FP32);
    auto* embed = embedding(index, vocab, "embed", axis);

    auto expected_shape = embed_output_shape(index_shape, vocab_shape);
    REQUIRE(embed != nullptr);
    REQUIRE(embed->has_producer());
    REQUIRE(embed->shape() == expected_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding backward", "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 100},
                   Index(2), Scalar(1.0)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 50},
                   Index(1), Scalar(-1.0)});

    NNGraph g("embedding_backward");
    auto* index = g.tensor(index_shape, "index", DataType::INT64, false);
    auto* vocab = g.tensor(vocab_shape, "vocab", DataType::FP32);
    auto* embed = embedding(index, vocab, "embed", axis);

    auto [embed_grad, _] = g.get_or_create_grad(embed, "embed_grad");
    fill(grad_fill_val, embed_grad);
    embed->backward();

    REQUIRE(vocab->has_grad());
    REQUIRE(vocab->grad()->shape() == vocab_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding forward and backward", "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 100},
                   Index(2), Scalar(1.0)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 50},
                   Index(1), Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 3, 4}, std::vector<Index>{6, 20},
                   Index(3), Scalar(-1.0)});

    NNGraph g("embedding");
    auto* index = g.tensor(index_shape, "index", DataType::INT64, false);
    auto* vocab = g.tensor(vocab_shape, "vocab", DataType::FP32, true);
    auto* embed = embedding(index, vocab, "embed", axis);

    auto expected_shape = embed_output_shape(index_shape, vocab_shape);
    REQUIRE(embed != nullptr);
    REQUIRE(embed->has_producer());
    REQUIRE(embed->shape() == expected_shape);

    auto [embed_grad, _] = g.get_or_create_grad(embed, "embed_grad");
    fill(grad_fill_val, embed_grad);
    embed->backward();

    REQUIRE(vocab->has_grad());
    REQUIRE(vocab->grad()->shape() == vocab_shape);
}
