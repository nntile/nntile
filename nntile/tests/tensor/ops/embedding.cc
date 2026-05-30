/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/embedding.cc
 * Test TensorGraph embedding operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/embedding.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/embedding.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

// embed_shape from index_shape, vocab_shape, axis
// embed.shape[axis] = vocab.shape[0], embed has index dims before/after axis
std::vector<Index> embed_output_shape(const std::vector<Index> &index_shape,
    const std::vector<Index> &vocab_shape,
    Index axis)
{
    std::vector<Index> embed_shape;
    embed_shape.reserve(index_shape.size() + 1);
    for (Index i = 0; i < axis; ++i)
    {
        embed_shape.push_back(index_shape[i]);
    }
    embed_shape.push_back(vocab_shape[0]);
    for (Index i = axis; i < static_cast<Index>(index_shape.size()); ++i)
    {
        embed_shape.push_back(index_shape[i]);
    }
    return embed_shape;
}

} 

TEST_CASE("TensorGraph embedding structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *index = graph.data({4, 5}, DataType::INT64)->set_name("index");
    auto *vocab = graph.data({10, 100})->set_name("vocab");
    auto *embed = graph.data({4, 5, 10})->set_name("embed");

    gt::embedding(index, vocab, embed, 2);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "EMBEDDING");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == embed);
}

TEST_CASE("TensorGraph embedding rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *index = graph.data({4, 5}, DataType::INT64)->set_name("index");
    auto *vocab = graph.data({10, 100})->set_name("vocab");
    auto *embed = graph.data({4, 5, 10})->set_name("embed");

    REQUIRE_THROWS_AS(
        gt::embedding(nullptr, vocab, embed, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::embedding(index, nullptr, embed, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::embedding(index, vocab, nullptr, 2), std::invalid_argument);
}

TEST_CASE("TensorGraph embedding with output_name", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *index = graph.data({4, 5}, DataType::INT64)->set_name("index");
    auto *vocab = graph.data({10, 100})->set_name("vocab");

    auto *embed = gt::embedding(index, vocab, 2)->set_name("embed");

    REQUIRE(embed != nullptr);
    // NNTile layout: embed.shape[axis] == vocab.shape[0] (embed_dim)
    REQUIRE(embed->shape() == std::vector<Index>{4, 5, 10});
    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph embedding tiled matches untiled",
    "[graph][tensor]")
{
    const auto [index_shape, vocab_shape, axis] = GENERATE(
        std::tuple{
            std::vector<Index>{4, 5}, std::vector<Index>{10, 100}, Index(2)},
        std::tuple{
            std::vector<Index>{3}, std::vector<Index>{8, 50}, Index(1)});

    const Index index_nelems = std::accumulate(
        index_shape.begin(), index_shape.end(), Index(1), std::multiplies<>());
    const Index vocab_nelems = std::accumulate(
        vocab_shape.begin(), vocab_shape.end(), Index(1), std::multiplies<>());

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> vocab_data(vocab_nelems);
    for (Index i = 0; i < index_nelems; ++i)
    {
        index_data[i] = static_cast<std::int64_t>(i % vocab_shape[1]);
    }
    for (Index i = 0; i < vocab_nelems; ++i)
    {
        vocab_data[i] = 0.1f * static_cast<float>(i % 7);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("embedding_untiled");
        auto *index_node =
            graph.data(index_shape, DataType::INT64)->set_name("index");
        auto *vocab_node =
            graph.data(vocab_shape, DataType::FP32)->set_name("vocab");
        index_node->mark_input(true);
        vocab_node->mark_input(true);

        auto *embed_node =
            gt::embedding(index_node, vocab_node, axis)->set_name("embed");
        embed_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(index_node, index_data);
        runtime.bind_data(vocab_node, vocab_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(embed_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("embedding_tiled");
        auto *index_node =
            graph.data(index_shape, DataType::INT64)->set_name("index");
        auto *vocab_node =
            graph.data(vocab_shape, DataType::FP32)->set_name("vocab");
        index_node->mark_input(true);
        vocab_node->mark_input(true);

        auto *embed_node =
            gt::embedding(index_node, vocab_node, axis)->set_name("embed");
        embed_node->mark_output(true);
        auto *num_embed_axis = vocab_node->axis(1);
        for (auto *ag : graph.axis_groups())
        {
            if (ag == num_embed_axis)
            {
                ag->set_tiling(ag->extent);
            }
            else
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(index_node, index_data);
        runtime.bind_data(vocab_node, vocab_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(embed_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
