/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/log_scalar.cc
 * Test TensorGraph log_scalar operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"
#include "nntile/graph/tensor/log_scalar.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tensor/log_scalar.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

TEST_CASE("TensorGraph log_scalar structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* value = graph.data({}, "value");
    gt::log_scalar("test_scalar", value);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "LOG_SCALAR");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().empty());
}

TEST_CASE("TensorGraph log_scalar rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");

    REQUIRE_THROWS_AS(gt::log_scalar("name", nullptr), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph log_scalar executes and preserves value", "[graph][tensor]")
{
    // log_scalar logs the value to the logger; we verify the tensor
    // is unchanged after execution (log_scalar is read-only on the tensor)
    TensorGraph graph("log_scalar_test");
    auto* value_node = graph.data({}, "value", DataType::FP32);
    value_node->mark_input(true);
    value_node->mark_output(true);

    gt::log_scalar("test_value", value_node);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();

    std::vector<float> value_data = {3.14f};
    runtime.bind_data("value", value_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> result = runtime.get_output<float>("value");
    REQUIRE(result.size() == 1);
    REQUIRE(std::abs(result[0] - 3.14f) < tolerance);
}
