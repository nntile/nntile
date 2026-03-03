/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/embedding.cc
 * Test TensorGraph embedding operation against nntile::tensor::embedding.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph/tensor/embedding.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/embedding.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

// embed_shape from index_shape, vocab_shape, axis
// embed.shape[axis] = vocab.shape[0], embed has index dims before/after axis
std::vector<Index> embed_output_shape(
    const std::vector<Index>& index_shape,
    const std::vector<Index>& vocab_shape,
    Index axis)
{
    std::vector<Index> embed_shape;
    embed_shape.reserve(index_shape.size() + 1);
    for(Index i = 0; i < axis; ++i)
    {
        embed_shape.push_back(index_shape[i]);
    }
    embed_shape.push_back(vocab_shape[0]);
    for(Index i = axis; i < static_cast<Index>(index_shape.size()); ++i)
    {
        embed_shape.push_back(index_shape[i]);
    }
    return embed_shape;
}

} // anonymous namespace

template<typename T>
void check_embedding_vs_tensor_api(
    const std::vector<Index>& index_shape,
    const std::vector<Index>& vocab_shape,
    Index axis)
{
    using Y = typename T::repr_t;
    auto embed_shape = embed_output_shape(index_shape, vocab_shape, axis);

    const Index index_nelems = std::accumulate(
        index_shape.begin(), index_shape.end(), Index(1), std::multiplies<>());
    const Index vocab_nelems = std::accumulate(
        vocab_shape.begin(), vocab_shape.end(), Index(1), std::multiplies<>());
    const Index embed_nelems = std::accumulate(
        embed_shape.begin(), embed_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("embedding_test");
    auto* index_node = graph.data(index_shape, "index", DataType::INT64);
    auto* vocab_node = graph.data(vocab_shape, "vocab", DataType::FP32);
    auto* embed_node = graph.data(embed_shape, "embed", DataType::FP32);
    index_node->mark_input(true);
    vocab_node->mark_input(true);
    embed_node->mark_input(true);
    embed_node->mark_output(true);

    embedding(index_node, vocab_node, embed_node, axis);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(vocab_nelems);
    std::vector<float> embed_data(embed_nelems, 0.0f);
    for(Index i = 0; i < index_nelems; ++i)
    {
        index_data[i] = static_cast<std::int64_t>(i % vocab_shape[1]);
    }
    for(Index i = 0; i < vocab_nelems; ++i)
    {
        vocab_data[i] = 0.1f * static_cast<float>(i % 7);
    }

    runtime.bind_data("index", index_data);
    runtime.bind_data("vocab", vocab_data);
    runtime.bind_data("embed", embed_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("embed");

    // --- Direct tensor API path ---
    tensor::TensorTraits index_traits(index_shape, index_shape);
    tensor::TensorTraits vocab_traits(vocab_shape, vocab_shape);
    tensor::TensorTraits embed_traits(embed_shape, embed_shape);
    std::vector<int> distr(1, distr_rank_single);

    tensor::Tensor<nntile::int64_t> index_t(index_traits, distr);
    tensor::Tensor<T> vocab_t(vocab_traits, distr);
    tensor::Tensor<T> embed_t(embed_traits, distr);

    auto init_index = [](tensor::Tensor<nntile::int64_t>& t,
                        const std::vector<std::int64_t>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = nntile::int64_t(data[i]);
        }
        loc.release();
    };
    auto init_float = [](tensor::Tensor<T>& t, const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = static_cast<Y>(data[i]);
        }
        loc.release();
    };

    init_index(index_t, index_data);
    init_float(vocab_t, vocab_data);

    tensor::embedding<T>(index_t, vocab_t, embed_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(embed_nelems);
    {
        auto tile = embed_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < embed_nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph embedding structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* index = graph.data({4, 5}, "index", DataType::INT64);
    auto* vocab = graph.data({10, 100}, "vocab");
    auto* embed = graph.data({4, 5, 10}, "embed");

    embedding(index, vocab, embed, 2);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "EMBEDDING");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == embed);
}

TEST_CASE("TensorGraph embedding rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* index = graph.data({4, 5}, "index", DataType::INT64);
    auto* vocab = graph.data({10, 100}, "vocab");
    auto* embed = graph.data({4, 5, 10}, "embed");

    REQUIRE_THROWS_AS(
        embedding(nullptr, vocab, embed, 2),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        embedding(index, nullptr, embed, 2),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        embedding(index, vocab, nullptr, 2),
        std::invalid_argument);
}

TEST_CASE("TensorGraph embedding with output_name", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* index = graph.data({4, 5}, "index", DataType::INT64);
    auto* vocab = graph.data({10, 100}, "vocab");

    auto* embed = embedding(index, vocab, "embed", 2);

    REQUIRE(embed != nullptr);
    // Graph uses embed_shape = index.shape + vocab.dim(1)
    REQUIRE(embed->shape() == std::vector<Index>{4, 5, 100});
    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph embedding matches tensor::embedding", "[graph][tensor]")
{
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{std::vector<Index>{4, 5}, std::vector<Index>{10, 100}, Index(2)},
        std::tuple{std::vector<Index>{3}, std::vector<Index>{8, 50}, Index(1)},
        std::tuple{std::vector<Index>{2, 3, 4}, std::vector<Index>{6, 20}, Index(3)});

    check_embedding_vs_tensor_api<nntile::fp32_t>(index_shape, vocab_shape, axis);
}
