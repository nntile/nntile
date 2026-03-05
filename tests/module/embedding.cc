/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/embedding.cc
 * Tests for Embedding module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/embedding.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Embedding ConstructorCreatesParameters", "[module]")
{
    NNGraph g("embedding");

    Embedding emb(g, "emb", 10, 100);
    REQUIRE(emb.vocab_tensor() != nullptr);
    REQUIRE(emb.vocab_tensor()->shape() == std::vector<Index>({100, 10}));
    REQUIRE(emb.vocab_tensor()->name() == "emb_vocab");
    REQUIRE(emb.parameters().size() == 1);
    REQUIRE(emb.num_embeddings() == 10);
    REQUIRE(emb.embed_dim() == 100);
}

TEST_CASE("Embedding ConstructorWithExistingTensor", "[module]")
{
    NNGraph g("embedding");

    // NNTile layout: vocab [embed_dim, num_embeddings]
    auto* vocab = g.tensor({50, 8}, "shared_vocab", DataType::FP32);

    Embedding emb(g, "emb", *vocab);
    REQUIRE(emb.vocab_tensor() == vocab);
    REQUIRE(emb.num_embeddings() == 8);
    REQUIRE(emb.embed_dim() == 50);
    REQUIRE(emb.parameters().size() == 1);
}

TEST_CASE("Embedding ConstructorValidations", "[module]")
{
    NNGraph g("embedding");

    auto* bad_vocab = g.tensor({10}, "bad_vocab", DataType::FP32);
    REQUIRE_THROWS_AS(
        Embedding(g, "emb", *bad_vocab),
        std::invalid_argument);

    auto* vocab_3d = g.tensor({2, 3, 4}, "vocab_3d", DataType::FP32);
    REQUIRE_THROWS_AS(
        Embedding(g, "emb", *vocab_3d),
        std::invalid_argument);
}

TEST_CASE("Embedding Callable", "[module]")
{
    NNGraph g("embedding_callable");
    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(g, "emb", 10, 100);
    auto& output = emb(*index);
    REQUIRE(output.shape() == std::vector<Index>({4, 5, 100}));
}

TEST_CASE("Embedding BuildForward", "[module]")
{
    NNGraph g("embedding");

    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(g, "emb", 10, 100);

    auto& output = emb.build_forward(*index);
    REQUIRE(output.shape() == std::vector<Index>({4, 5, 100}));
    REQUIRE(output.name() == "emb_output");
    REQUIRE(g.num_ops() >= 1);
    REQUIRE(output.has_producer());
}

TEST_CASE("Embedding BuildForwardValidatesIndexDtype", "[module]")
{
    NNGraph g("embedding");

    auto* bad_index = g.tensor({4, 5}, "bad_index", DataType::FP32);
    Embedding emb(g, "emb", 10, 100);

    REQUIRE_THROWS_AS(
        emb.build_forward(*bad_index),
        std::invalid_argument);
}

TEST_CASE("Embedding BuildForwardRejectsScalarIndex", "[module]")
{
    NNGraph g("embedding");

    auto* scalar = g.tensor({}, "scalar", DataType::INT64, false);
    Embedding emb(g, "emb", 10, 100);

    REQUIRE_THROWS_AS(
        emb.build_forward(*scalar),
        std::invalid_argument);
}

TEST_CASE("Embedding BackwardCreatesGradients", "[module]")
{
    NNGraph g("embedding");

    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(g, "emb", 10, 100);

    auto& output = emb.build_forward(*index);
    g.get_or_create_grad(&output, "output_grad");
    output.backward();

    REQUIRE(emb.vocab_tensor()->grad() != nullptr);
    REQUIRE(emb.vocab_tensor()->grad()->shape() ==
        std::vector<Index>({100, 10}));
}
