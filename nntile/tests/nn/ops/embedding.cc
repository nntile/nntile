/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/embedding.cc
 * Test NNGraph embedding autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile;

// Virtual graph: insert embed_dim at ``c_axis`` (default append).
static std::vector<Index> embed_output_shape(
    const std::vector<Index> &index_shape,
    const std::vector<Index> &vocab_shape,
    Index c_axis = -1)
{
    const Index index_ndim = static_cast<Index>(index_shape.size());
    const Index axis = (c_axis < 0) ? index_ndim : c_axis;
    std::vector<Index> shape(index_shape.begin(), index_shape.end());
    shape.insert(shape.begin() + axis, vocab_shape.back());
    return shape;
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding structure",
    "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{
            std::vector<Index>{5, 4}, std::vector<Index>{100, 10}, Index(2)},
        std::tuple{
            std::vector<Index>{3}, std::vector<Index>{50, 8}, Index(1)});

    NNGraph g("embedding_structure");
    auto *index =
        g.tensor(index_shape, DataType::INT64, false)->set_name("index");
    auto *vocab = g.tensor(vocab_shape, DataType::FP32)->set_name("vocab");
    auto *embed = embedding(index, vocab, axis)->set_name("embed");

    auto expected_shape =
        embed_output_shape(index_shape, vocab_shape, axis);
    REQUIRE(embed != nullptr);
    REQUIRE(embed->has_producer());
    REQUIRE(embed->shape() == expected_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding backward",
    "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{5, 4},
                     std::vector<Index>{100, 10},
                     Index(2),
                     Scalar(1.0)},
            std::tuple{std::vector<Index>{3},
                std::vector<Index>{50, 8},
                Index(1),
                Scalar(-1.0)});

    NNGraph g("embedding_backward");
    auto *index =
        g.tensor(index_shape, DataType::INT64, false)->set_name("index");
    auto *vocab = g.tensor(vocab_shape, DataType::FP32)->set_name("vocab");
    auto *embed = embedding(index, vocab, axis)->set_name("embed");

    auto [embed_grad, _] = g.get_or_create_grad(embed, "embed_grad");
    fill(grad_fill_val, embed_grad);
    embed->backward();

    REQUIRE(vocab->has_grad());
    REQUIRE(vocab->grad()->shape() == vocab_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding forward and backward",
    "[graph][nn_graph]")
{
    const auto [index_shape, vocab_shape, axis, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{5, 4},
                     std::vector<Index>{100, 10},
                     Index(2),
                     Scalar(1.0)},
            std::tuple{std::vector<Index>{3},
                std::vector<Index>{50, 8},
                Index(1),
                Scalar(1.0)},
            std::tuple{std::vector<Index>{4, 3, 2},
                std::vector<Index>{20, 6},
                Index(3),
                Scalar(-1.0)});

    NNGraph g("embedding");
    auto *index =
        g.tensor(index_shape, DataType::INT64, false)->set_name("index");
    auto *vocab =
        g.tensor(vocab_shape, DataType::FP32, true)->set_name("vocab");
    auto *embed = embedding(index, vocab, axis)->set_name("embed");

    auto expected_shape =
        embed_output_shape(index_shape, vocab_shape, axis);
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

using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_vocab_10x10;
using nntile::test::nn_pytorch_tile_vocab_8x8;

static std::vector<std::int64_t> embedding_pytorch_output_shape(
    const std::vector<Index> &index_shape, Index embed_dim)
{
    std::vector<std::int64_t> shape(index_shape.begin(), index_shape.end());
    shape.push_back(static_cast<std::int64_t>(embed_dim));
    return shape;
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding forward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    // Square vocab keeps embed_dim == num_embeddings for PyTorch parity.
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{
            std::vector<Index>{5, 4}, std::vector<Index>{10, 10}, Index(2)});

    const Index embed_dim = vocab_shape.back();
    const Index num_embeddings = vocab_shape.front();
    Index index_nelems = 1;
    for (Index d : index_shape)
        index_nelems *= d;

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(embed_dim * num_embeddings);
    for (Index i = 0; i < index_nelems; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    for (Index i = 0; i < embed_dim * num_embeddings; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("embedding_pytorch");
    auto *index =
        g.tensor(index_shape, DataType::INT64, false)->set_name("index");
    auto *vocab =
        g.tensor(vocab_shape, DataType::FP32, true)->set_name("vocab");
    auto *embed = embedding(index, vocab, axis)->set_name("embed");

    nn_pytorch_tile_vocab_10x10(vocab);

    index->mark_input(true);
    vocab->mark_input(true);
    embed->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(index, index_data);
    runtime.bind_data(vocab, vocab_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(embed);

    std::vector<::int64_t> index_shape_pt(
        index_shape.begin(), index_shape.end());
    auto index_pt = torch::from_blob(index_data.data(),
        index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64))
                        .clone()
                        .set_requires_grad(false);
    auto vocab_pt = torch::from_blob(vocab_data.data(),
        {static_cast<long>(num_embeddings), static_cast<long>(embed_dim)},
        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(false);
    auto out_pt = torch::embedding(vocab_pt, index_pt).contiguous();
    compare_float_vectors(nntile_out, out_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph embedding backward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    // Square vocab required: embed.shape[axis]==vocab.shape.back()
    const auto [index_shape, vocab_shape, axis, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{5, 4},
                     std::vector<Index>{10, 10},
                     Index(2),
                     Scalar(1.0)},
            std::tuple{std::vector<Index>{3},
                std::vector<Index>{8, 8},
                Index(1),
                Scalar(-1.0)});

    const Index embed_dim = vocab_shape.back();
    const Index num_embeddings = vocab_shape.front();
    Index index_nelems = 1;
    for (Index d : index_shape)
        index_nelems *= d;

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(num_embeddings * embed_dim);
    for (Index i = 0; i < index_nelems; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    for (Index i = 0; i < num_embeddings * embed_dim; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("embedding_bwd_pytorch");
    auto *index =
        g.tensor(index_shape, DataType::INT64, false)->set_name("index");
    auto *vocab =
        g.tensor(vocab_shape, DataType::FP32, true)->set_name("vocab");
    auto *embed = embedding(index, vocab, axis)->set_name("embed");

    if (index_shape.size() == 2)
        nn_pytorch_tile_vocab_10x10(vocab);
    else
        nn_pytorch_tile_vocab_8x8(vocab);

    index->mark_input(true);
    vocab->mark_input(true);

    auto [embed_grad, _] = g.get_or_create_grad(embed, "embed_grad");
    fill(grad_fill_val, embed_grad);
    embed->backward();

    vocab->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(index, index_data);
    runtime.bind_data(vocab, vocab_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_vocab =
        runtime.get_output<float>(vocab->grad());

    std::vector<::int64_t> index_shape_pt(
        index_shape.begin(), index_shape.end());
    auto index_pt = torch::from_blob(index_data.data(),
        index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64))
                        .clone()
                        .set_requires_grad(false);
    auto vocab_pt = torch::from_blob(vocab_data.data(),
        {static_cast<long>(num_embeddings), static_cast<long>(embed_dim)},
        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(true);
    auto out_pt = torch::embedding(vocab_pt, index_pt);

    const std::vector<std::int64_t> out_shape_pt =
        embedding_pytorch_output_shape(index_shape, embed_dim);
    auto grad_output = torch::full(out_shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_vocab, vocab_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
