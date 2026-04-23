/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/embedding.cc
 * Tests for Embedding module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/modules/embedding.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/graph/module/embedding.hh"
#include "nntile/graph/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#   include "pytorch_tile_helpers.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;

TEST_CASE("Embedding ConstructorCreatesParameters", "[module]")
{
    NNGraph g("embedding");

    Embedding emb(&g, "emb", 10, 100);
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

    Embedding emb(&g, "emb", vocab);
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
        Embedding(&g, "emb", bad_vocab),
        std::invalid_argument);

    auto* vocab_3d = g.tensor({2, 3, 4}, "vocab_3d", DataType::FP32);
    REQUIRE_THROWS_AS(
        Embedding(&g, "emb", vocab_3d),
        std::invalid_argument);
}

TEST_CASE("Embedding Callable", "[module]")
{
    NNGraph g("embedding_callable");
    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", 10, 100);
    auto* output = emb(index);
    REQUIRE(output->shape() == std::vector<Index>({4, 5, 100}));
}

TEST_CASE("Embedding BuildForward", "[module]")
{
    NNGraph g("embedding");

    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", 10, 100);

    auto* output = emb.forward(index);
    REQUIRE(output->shape() == std::vector<Index>({4, 5, 100}));
    REQUIRE(output->name() == "emb_output");
    REQUIRE(g.num_ops() >= 1);
    REQUIRE(output->has_producer());
}

TEST_CASE("Embedding BuildForwardValidatesIndexDtype", "[module]")
{
    NNGraph g("embedding");

    auto* bad_index = g.tensor({4, 5}, "bad_index", DataType::FP32);
    Embedding emb(&g, "emb", 10, 100);

    REQUIRE_THROWS_AS(
        emb.forward(bad_index),
        std::invalid_argument);
}

TEST_CASE("Embedding BuildForwardRejectsScalarIndex", "[module]")
{
    NNGraph g("embedding");

    auto* scalar = g.tensor({}, "scalar", DataType::INT64, false);
    Embedding emb(&g, "emb", 10, 100);

    REQUIRE_THROWS_AS(
        emb.forward(scalar),
        std::invalid_argument);
}

TEST_CASE("Embedding BackwardCreatesGradients", "[module]")
{
    NNGraph g("embedding");

    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", 10, 100);

    auto* output = emb.forward(index);
    g.get_or_create_grad(output, "output_grad");
    output->backward();

    REQUIRE(emb.vocab_tensor()->grad() != nullptr);
    REQUIRE(emb.vocab_tensor()->grad()->shape() ==
        std::vector<Index>({100, 10}));
}

#ifdef NNTILE_HAVE_TORCH

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding bind_weight applies data on compile", "[module]")
{
    const Index num_embeddings = 10;
    const Index embed_dim = 100;

    NNGraph g("embedding_bind");
    auto* index = g.tensor({4, 5}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", num_embeddings, embed_dim);

    auto* output = emb.forward(index);
    index->mark_input(true);
    output->mark_output(true);

    // Bind vocab before compile; data in NNTile (column-major) layout
    // vocab shape [embed_dim, num_embeddings]
    std::vector<float> vocab_data(embed_dim * num_embeddings);
    for(Index i = 0; i < embed_dim * num_embeddings; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);
    emb.bind_weight(vocab_data);

    nntile::test::module_apply_embedding_vocab_tiling(emb.vocab_tensor());
    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<std::int64_t> index_data(4 * 5);
    for(Index i = 0; i < 20; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    runtime.bind_data(index->name(), index_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == 4 * 5 * embed_dim);
    // output[0,0,:] = vocab[:, index[0,0]]. index[0,0]=0, so first column of vocab
    // In col-major vocab, column 0 is vocab[0:embed_dim]
    // Output layout [batch, seq_len, embed_dim]: stride for embed_dim is 4*5=20
    const Index stride = 4 * 5;
    for(Index i = 0; i < embed_dim; ++i)
        REQUIRE(std::abs(out[i * stride] - vocab_data[i]) < 1e-5f);
}

using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding from PyTorch binds weight in constructor", "[module][pytorch]")
{
    const Index num_embeddings = 10;
    const Index embed_dim = 100;
    const Index batch = 4;
    const Index seq_len = 5;

    torch::manual_seed(42);
    auto emb_pt = torch::nn::Embedding(num_embeddings, embed_dim);

    NNGraph g("embedding_from_pytorch");
    auto* index = g.tensor({batch, seq_len}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", emb_pt);
    auto* output = emb.forward(index);

    index->mark_input(true);
    output->mark_output(true);

    std::vector<std::int64_t> index_data(batch * seq_len);
    for(Index i = 0; i < batch * seq_len; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);

    nntile::test::module_apply_embedding_vocab_tiling(emb.vocab_tensor());
    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("index", index_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output->name());
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, seq_len, embed_dim});

    // Index data is column-major (NNTile layout); convert for PyTorch row-major
    std::vector<std::int64_t> index_rowmajor =
        colmajor_to_rowmajor(index_data, {batch, seq_len});
    std::vector<std::int64_t> index_shape_pt{batch, seq_len};
    auto index_pt = torch::from_blob(index_rowmajor.data(), index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto out_pt = emb_pt->forward(index_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * seq_len * embed_dim);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding from PyTorch forward-backward", "[module][pytorch]")
{
    const auto [num_embeddings, embed_dim, batch, seq_len] = GENERATE(
        std::tuple{Index(10), Index(100), Index(4), Index(5)},
        std::tuple{Index(8), Index(50), Index(2), Index(3)});

    torch::manual_seed(42);
    auto emb_pt = torch::nn::Embedding(num_embeddings, embed_dim);

    std::vector<std::int64_t> index_data(batch * seq_len);
    for(Index i = 0; i < batch * seq_len; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);

    NNGraph g("embedding_fwd_bwd_pytorch");
    auto* index = g.tensor({batch, seq_len}, "index", DataType::INT64, false);
    Embedding emb(&g, "emb", emb_pt);
    auto* output = emb.forward(index);

    index->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    fill(Scalar(1.0f), grad_output_tensor);
    output->backward();

    emb.vocab_tensor()->grad()->mark_output(true);

    nntile::test::module_apply_embedding_vocab_tiling(emb.vocab_tensor());
    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("index", index_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output->name());
    REQUIRE(out.size() == static_cast<size_t>(batch * seq_len * embed_dim));

    auto grad_vocab = runtime.get_output<float>(emb.grad_name("vocab"));
    REQUIRE(grad_vocab.size() ==
        static_cast<size_t>(embed_dim * num_embeddings));
}

#endif // NNTILE_HAVE_TORCH
