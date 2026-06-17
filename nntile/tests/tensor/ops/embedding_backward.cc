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

// embed_shape from index_shape, vocab_shape, axis
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
    embed_shape.push_back(vocab_shape.back());
    for (Index i = axis; i < static_cast<Index>(index_shape.size()); ++i)
    {
        embed_shape.push_back(index_shape[i]);
    }
    return embed_shape;
}

} 

TEST_CASE("TensorGraph embedding_backward structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *index = graph.data({5, 4}, DataType::INT64)->set_name("index");
    auto *embed = graph.data({5, 4, 10})->set_name("embed");
    auto *vocab = graph.data({100, 10})->set_name("vocab");

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
    auto *index = graph.data({5, 4}, DataType::INT64)->set_name("index");
    auto *embed = graph.data({5, 4, 10})->set_name("embed");
    auto *vocab = graph.data({100, 10})->set_name("vocab");

    REQUIRE_THROWS_AS(gt::embedding_backward(nullptr, embed, vocab, 2, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::embedding_backward(index, nullptr, vocab, 2, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::embedding_backward(index, embed, nullptr, 2, 0),
        std::invalid_argument);
}
