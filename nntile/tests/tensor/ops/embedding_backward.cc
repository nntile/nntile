/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/embedding_backward.cc
 * Test TensorGraph embedding_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/embedding_backward.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/embedding_backward.hh"
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

// embed_shape: index_shape + [vocab embed_dim] at the last axis (axis ==
// index.ndim)
std::vector<Index> embed_output_shape(const std::vector<Index> &index_shape,
    const std::vector<Index> &vocab_shape)
{
    std::vector<Index> embed_shape = index_shape;
    embed_shape.push_back(vocab_shape[1]);
    return embed_shape;
}

} 

TEST_CASE("TensorGraph embedding_backward structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef index = graph.data({5, 4}, DataType::INT64);
    index->set_name("index");
    nntile::TensorRef embed = graph.data({5, 4, 100});
    embed->set_name("embed");
    nntile::TensorRef vocab = graph.data({10, 100});
    vocab->set_name("vocab");

    gt::embedding_backward(index, embed, vocab, 2, 0);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "EMBEDDING_BACKWARD");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == vocab);
}

TEST_CASE(
    "TensorGraph embedding_backward rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef index = graph.data({5, 4}, DataType::INT64);
    index->set_name("index");
    nntile::TensorRef embed = graph.data({5, 4, 100});
    embed->set_name("embed");
    nntile::TensorRef vocab = graph.data({10, 100});
    vocab->set_name("vocab");

    REQUIRE_THROWS_AS(gt::embedding_backward(nullptr, embed, vocab, 2, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::embedding_backward(index, nullptr, vocab, 2, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::embedding_backward(index, embed, nullptr, 2, 0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph embedding_backward tiled matches untiled",
    "[graph][tensor]")
{
    const auto [index_shape, vocab_shape, axis, redux] = GENERATE(
        std::tuple{std::vector<Index>{5, 4},
            std::vector<Index>{10, 100},
            Index(2),
            0},
        std::tuple{
            std::vector<Index>{3}, std::vector<Index>{8, 50}, Index(1), 0});

    auto embed_shape = embed_output_shape(index_shape, vocab_shape);

    const Index index_nelems = std::accumulate(
        index_shape.begin(), index_shape.end(), Index(1), std::multiplies<>());
    const Index embed_nelems = std::accumulate(
        embed_shape.begin(), embed_shape.end(), Index(1), std::multiplies<>());
    const Index vocab_nelems = std::accumulate(
        vocab_shape.begin(), vocab_shape.end(), Index(1), std::multiplies<>());

    std::vector<std::int64_t> index_data(index_nelems);
    std::vector<float> embed_data(embed_nelems);
    std::vector<float> vocab_data(vocab_nelems, 0.0f);
    for (Index i = 0; i < index_nelems; ++i)
    {
        index_data[i] = static_cast<std::int64_t>(i % vocab_shape[0]);
    }
    for (Index i = 0; i < embed_nelems; ++i)
    {
        embed_data[i] = 0.1f * static_cast<float>(i % 5);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("embedding_backward_untiled");
        nntile::TensorRef index_node = graph.data(index_shape, DataType::INT64);
    index_node->set_name("index");
        nntile::TensorRef embed_node = graph.data(embed_shape, DataType::FP32);
    embed_node->set_name("embed");
        nntile::TensorRef vocab_node = graph.data(vocab_shape, DataType::FP32);
    vocab_node->set_name("vocab");

        gt::embedding_backward(
            index_node, embed_node, vocab_node, axis, redux);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(index_node, index_data);
        runtime.bind_data(embed_node, embed_data);
        runtime.bind_data(vocab_node, vocab_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(vocab_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("embedding_backward_tiled");
        nntile::TensorRef index_node = graph.data(index_shape, DataType::INT64);
    index_node->set_name("index");
        nntile::TensorRef embed_node = graph.data(embed_shape, DataType::FP32);
    embed_node->set_name("embed");
        nntile::TensorRef vocab_node = graph.data(vocab_shape, DataType::FP32);
    vocab_node->set_name("vocab");

        gt::embedding_backward(
            index_node, embed_node, vocab_node, axis, redux);
        auto *embed_dim_axis = vocab_node->axis(1);
        auto *num_embeddings_axis = vocab_node->axis(0);
        for (auto *ag : graph.axis_groups())
        {
            if (ag == embed_dim_axis || ag == num_embeddings_axis)
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
        runtime.bind_data(embed_node, embed_data);
        runtime.bind_data(vocab_node, vocab_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(vocab_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
