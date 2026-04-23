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

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include "pytorch_tile_helpers.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

// NNTile layout: vocab [embed_dim, num_embeddings]; embed.shape[axis] == vocab.shape[0]
static std::vector<Index> embed_output_shape(
    const std::vector<Index>& index_shape,
    const std::vector<Index>& vocab_shape)
{
    std::vector<Index> embed_shape = index_shape;
    embed_shape.push_back(vocab_shape[0]);
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

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_index_4x5;
using nntile::test::nn_pytorch_tile_index_len3;
using nntile::test::nn_pytorch_tile_vocab_10x10;
using nntile::test::nn_pytorch_tile_vocab_8x8;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    // NNTile tensor embedding requires embed.shape[axis]==vocab.shape[0];
    // NNGraph creates embed with shape index_shape + [vocab.shape[0]] at the
    // last dim (axis == index.ndim). Square vocab keeps embed_dim ==
    // num_embeddings for a simple PyTorch `nn.Embedding` weight layout.
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 10},
                   Index(2)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 8}, Index(1)});

    const Index embed_dim = vocab_shape[0];
    const Index num_embeddings = vocab_shape[1];
    Index index_nelems = 1;
    for(Index d : index_shape)
        index_nelems *= d;

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(embed_dim * num_embeddings);
    for(Index i = 0; i < index_nelems; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    for(Index i = 0; i < embed_dim * num_embeddings; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<std::int64_t> index_rowmajor =
        colmajor_to_rowmajor(index_data, index_shape);
    std::vector<float> vocab_rowmajor =
        colmajor_to_rowmajor(vocab_data, vocab_shape);

    std::vector<Index> embed_shape = index_shape;
    embed_shape.push_back(embed_dim);
    const Index embed_nelems = index_nelems * embed_dim;

    NNGraph g("embedding_pytorch");
    auto* index = g.tensor(index_shape, "index", DataType::INT64, false);
    auto* vocab = g.tensor(vocab_shape, "vocab", DataType::FP32, true);
    auto* embed = embedding(index, vocab, "embed", axis);

    if(index_shape.size() == 2)
    {
        nn_pytorch_tile_index_4x5(index);
        nn_pytorch_tile_vocab_10x10(vocab);
    }
    else
    {
        nn_pytorch_tile_index_len3(index);
        nn_pytorch_tile_vocab_8x8(vocab);
    }

    index->mark_input(true);
    vocab->mark_input(true);
    embed->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("index", index_data);
    runtime.bind_data("vocab", vocab_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("embed");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, embed_shape);

    // PyTorch weight: (num_embeddings, embed_dim); NNTile vocab: (embed_dim, num_embeddings)
    std::vector<::int64_t> index_shape_pt(index_shape.begin(), index_shape.end());
    auto index_pt = torch::from_blob(index_rowmajor.data(), index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64)).clone().set_requires_grad(false);
    auto vocab_pt = torch::from_blob(vocab_rowmajor.data(),
        {static_cast<long>(embed_dim), static_cast<long>(num_embeddings)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().t().contiguous();
    vocab_pt.set_requires_grad(false);
    auto out_pt = torch::embedding(vocab_pt, index_pt).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + embed_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    // Square vocab required: embed.shape[axis]==vocab.shape[0]==vocab.shape[1]
    const auto [index_shape, vocab_shape, axis, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 10},
                   Index(2), Scalar(1.0)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 8},
                   Index(1), Scalar(-1.0)});

    const Index embed_dim = vocab_shape[0];
    const Index num_embeddings = vocab_shape[1];
    Index index_nelems = 1;
    for(Index d : index_shape)
        index_nelems *= d;

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(num_embeddings * embed_dim);
    for(Index i = 0; i < index_nelems; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    for(Index i = 0; i < num_embeddings * embed_dim; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<std::int64_t> index_rowmajor =
        colmajor_to_rowmajor(index_data, index_shape);
    std::vector<float> vocab_rowmajor =
        colmajor_to_rowmajor(vocab_data, vocab_shape);

    std::vector<Index> embed_shape = index_shape;
    embed_shape.push_back(embed_dim);

    NNGraph g("embedding_bwd_pytorch");
    auto* index = g.tensor(index_shape, "index", DataType::INT64, false);
    auto* vocab = g.tensor(vocab_shape, "vocab", DataType::FP32, true);
    auto* embed = embedding(index, vocab, "embed", axis);

    if(index_shape.size() == 2)
    {
        nn_pytorch_tile_index_4x5(index);
        nn_pytorch_tile_vocab_10x10(vocab);
    }
    else
    {
        nn_pytorch_tile_index_len3(index);
        nn_pytorch_tile_vocab_8x8(vocab);
    }

    index->mark_input(true);
    vocab->mark_input(true);

    auto [embed_grad, _] = g.get_or_create_grad(embed, "embed_grad");
    fill(grad_fill_val, embed_grad);
    embed->backward();

    vocab->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("index", index_data);
    runtime.bind_data("vocab", vocab_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_vocab_colmajor =
        runtime.get_output<float>(vocab->grad()->name());
    std::vector<float> nntile_grad_vocab =
        colmajor_to_rowmajor(nntile_grad_vocab_colmajor, vocab_shape);

    std::vector<::int64_t> index_shape_pt(index_shape.begin(), index_shape.end());
    auto index_pt = torch::from_blob(index_rowmajor.data(), index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64)).clone().set_requires_grad(false);
    auto vocab_pt = torch::from_blob(vocab_rowmajor.data(),
        {static_cast<long>(embed_dim), static_cast<long>(num_embeddings)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().t().contiguous();
    vocab_pt.set_requires_grad(true);
    auto out_pt = torch::embedding(vocab_pt, index_pt);

    std::vector<::int64_t> embed_shape_pt(embed_shape.begin(), embed_shape.end());
    auto grad_output = torch::full(embed_shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    // NNTile vocab: (embed_dim, num_embeddings); PyTorch weight: (num_embeddings, embed_dim)
    // grad_vocab[i,j] = grad_weight[j,i], so transpose for comparison
    std::vector<float> nntile_grad_for_compare(num_embeddings * embed_dim);
    for(Index i = 0; i < embed_dim; ++i)
        for(Index j = 0; j < num_embeddings; ++j)
            nntile_grad_for_compare[j * embed_dim + i] =
                nntile_grad_vocab[i * num_embeddings + j];
    compare_float_vectors(nntile_grad_for_compare, vocab_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
