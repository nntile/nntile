/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/module/embedding.cc
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
#include <torch/nn/modules/embedding.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/embedding.hh"
#include "nntile/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#include "context_fixture.hh"
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

using namespace nntile;
using namespace nntile;
using namespace nntile::module;

static std::vector<Index> embed_output_shape(
    const std::vector<Index> &index_shape,
    Index embed_dim,
    Index c_axis = -1)
{
    const Index index_ndim = static_cast<Index>(index_shape.size());
    const Index axis = (c_axis < 0) ? index_ndim : c_axis;
    std::vector<Index> shape(index_shape.begin(), index_shape.end());
    shape.insert(shape.begin() + axis, embed_dim);
    return shape;
}

TEST_CASE("Embedding ConstructorCreatesParameters", "[module]")
{
    NNGraph g("embedding");

    Embedding emb(&g, "emb", 10, 100);
    REQUIRE(emb.vocab_tensor() != nullptr);
    REQUIRE(emb.vocab_tensor()->shape() == std::vector<Index>({10, 100}));
    REQUIRE(emb.vocab_tensor()->name() == "emb_vocab");
    REQUIRE(emb.parameters().size() == 1);
    REQUIRE(emb.num_embeddings() == 10);
    REQUIRE(emb.embed_dim() == 100);
}

TEST_CASE("Embedding ConstructorWithExistingTensor", "[module]")
{
    NNGraph g("embedding");

    // graph vocab [num_embeddings, embed_dim]
    auto *vocab = g.tensor({8, 50}, DataType::FP32)->set_name("shared_vocab");

    Embedding emb(&g, "emb", vocab);
    REQUIRE(emb.vocab_tensor() == vocab);
    REQUIRE(emb.num_embeddings() == 8);
    REQUIRE(emb.embed_dim() == 50);
    REQUIRE(emb.parameters().size() == 1);
}

TEST_CASE("Embedding ConstructorValidations", "[module]")
{
    NNGraph g("embedding");

    auto *bad_vocab = g.tensor({10}, DataType::FP32)->set_name("bad_vocab");
    REQUIRE_THROWS_AS(Embedding(&g, "emb", bad_vocab), std::invalid_argument);

    auto *vocab_3d = g.tensor({2, 3, 4}, DataType::FP32)->set_name("vocab_3d");
    REQUIRE_THROWS_AS(Embedding(&g, "emb", vocab_3d), std::invalid_argument);
}

TEST_CASE("Embedding Callable", "[module]")
{
    NNGraph g("embedding_callable");
    auto *index = g.tensor({4, 5}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", 10, 100);
    auto *output = emb(index);
    REQUIRE(output->shape() == embed_output_shape({4, 5}, 100));
}

TEST_CASE("Embedding BuildForward", "[module]")
{
    NNGraph g("embedding");

    auto *index = g.tensor({4, 5}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", 10, 100);

    auto *output = emb.forward(index);
    REQUIRE(output->shape() == embed_output_shape({4, 5}, 100));
    REQUIRE(output->name() == "emb_output");
    REQUIRE(g.num_ops() >= 1);
    REQUIRE(output->has_producer());
}

TEST_CASE("Embedding BuildForwardValidatesIndexDtype", "[module]")
{
    NNGraph g("embedding");

    auto *bad_index = g.tensor({4, 5}, DataType::FP32)->set_name("bad_index");
    Embedding emb(&g, "emb", 10, 100);

    REQUIRE_THROWS_AS(emb.forward(bad_index), std::invalid_argument);
}

TEST_CASE("Embedding BuildForwardRejectsScalarIndex", "[module]")
{
    NNGraph g("embedding");

    auto *scalar = g.tensor({}, DataType::INT64, false)->set_name("scalar");
    Embedding emb(&g, "emb", 10, 100);

    REQUIRE_THROWS_AS(emb.forward(scalar), std::invalid_argument);
}

TEST_CASE("Embedding BackwardCreatesGradients", "[module]")
{
    NNGraph g("embedding");

    auto *index = g.tensor({4, 5}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", 10, 100);

    auto *output = emb.forward(index);
    g.get_or_create_grad(output, "output_grad");
    output->backward();

    REQUIRE(emb.vocab_tensor()->grad() != nullptr);
    REQUIRE(
        emb.vocab_tensor()->grad()->shape() == std::vector<Index>({10, 100}));
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding bind_weight applies data on compile",
    "[module]")
{
    const Index num_embeddings = 10;
    const Index embed_dim = 100;

    NNGraph g("embedding_bind");
    auto *index = g.tensor({4, 5}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", num_embeddings, embed_dim);

    auto *output = emb.forward(index);
    index->mark_input(true);
    output->mark_output(true);

    // Bind vocab before compile; data in [num_embeddings, embed_dim] layout
    // vocab shape [num_embeddings, embed_dim]
    std::vector<float> vocab_data(num_embeddings * embed_dim);
    for (Index i = 0; i < num_embeddings * embed_dim; ++i)
        vocab_data[i] = 0.1f * static_cast<float>(i + 1);
    emb.bind_weight(vocab_data);

    nntile::test::module_apply_embedding_vocab_tiling(emb.vocab_tensor());
    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());

    Runtime runtime(tile_graph);
    runtime.compile();

    g.bind_parameters(runtime);

    std::vector<std::int64_t> index_data(4 * 5);
    for (Index i = 0; i < 20; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);
    runtime.bind_data(index, index_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output);
    REQUIRE(out.size() == 4 * 5 * embed_dim);
    std::vector<std::int64_t> index_shape_pt{4, 5};
    auto index_pt = torch::from_blob(index_data.data(),
        index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64))
                        .clone();
    auto vocab_pt = torch::from_blob(vocab_data.data(),
        {static_cast<long>(num_embeddings), static_cast<long>(embed_dim)},
        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone();
    auto out_pt = torch::embedding(vocab_pt, index_pt).contiguous();
    compare_float_vectors(out, out_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding from PyTorch binds weight in constructor",
    "[module][pytorch]")
{
    const Index num_embeddings = 10;
    const Index embed_dim = 100;
    const Index batch = 4;
    const Index seq_len = 5;

    torch::manual_seed(42);
    auto emb_pt = torch::nn::Embedding(num_embeddings, embed_dim);

    NNGraph g("embedding_from_pytorch");
    auto *index =
        g.tensor({batch, seq_len}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", emb_pt);
    auto *output = emb.forward(index);

    index->mark_input(true);
    output->mark_output(true);

    std::vector<std::int64_t> index_data(batch * seq_len);
    for (Index i = 0; i < batch * seq_len; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);

    nntile::test::module_apply_embedding_vocab_tiling(emb.vocab_tensor());
    nntile::test::module_tile_all_untiled_axis_groups_heterogeneous(
        g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());

    Runtime runtime(tile_graph);
    runtime.compile();
    g.bind_parameters(runtime);
    runtime.bind_data(index, index_data);
    runtime.execute();
    runtime.wait();

    std::vector<std::int64_t> index_shape_pt{batch, seq_len};
    auto index_pt = torch::from_blob(index_data.data(),
        index_shape_pt,
        torch::TensorOptions().dtype(torch::kInt64))
                        .clone();
    auto out_pt = emb_pt->forward(index_pt);
    compare_float_vectors(runtime.get_output<float>(output), out_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Embedding from PyTorch forward-backward",
    "[module][pytorch]")
{
    const auto [num_embeddings, embed_dim, batch, seq_len] =
        GENERATE(std::tuple{Index(10), Index(100), Index(4), Index(5)},
            std::tuple{Index(8), Index(50), Index(2), Index(3)});

    torch::manual_seed(42);
    auto emb_pt = torch::nn::Embedding(num_embeddings, embed_dim);

    std::vector<std::int64_t> index_data(batch * seq_len);
    for (Index i = 0; i < batch * seq_len; ++i)
        index_data[i] = static_cast<std::int64_t>(i % num_embeddings);

    NNGraph g("embedding_fwd_bwd_pytorch");
    auto *index =
        g.tensor({batch, seq_len}, DataType::INT64, false)->set_name("index");
    Embedding emb(&g, "emb", emb_pt);
    auto *output = emb.forward(index);

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

    Runtime runtime(tile_graph);
    runtime.compile();
    g.bind_parameters(runtime);
    runtime.bind_data(index, index_data);
    runtime.execute();
    runtime.wait();

    auto out = runtime.get_output<float>(output);
    REQUIRE(out.size() == static_cast<size_t>(batch * seq_len * embed_dim));

    auto grad_vocab = runtime.get_output<float>(emb.vocab_tensor()->grad());
    REQUIRE(
        grad_vocab.size() == static_cast<size_t>(num_embeddings * embed_dim));
}

#endif // NNTILE_HAVE_TORCH
