/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/tensor/ops/unregister.cc
 * Test TensorGraph unregister structure and unmarked-phase skip.
 */

#include "nntile/tensor/ops/unregister.hh"

#include "nntile/tensor.hh"
#include "nntile/tensor/ops/invalidate.hh"

#include <catch2/catch_test_macros.hpp>

using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph unregister structure", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef x = graph.data({4, 5});
    x->set_name("x");
    gt::unregister(x);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "UNREGISTER");
    REQUIRE(graph.ops()[0]->inputs().size() == 1);
    REQUIRE(graph.ops()[0]->inputs()[0] == x);
    REQUIRE(graph.ops()[0]->outputs().empty());
}

TEST_CASE(
    "append_invalidates_for_unmarked_unsealed skips UNREGISTER",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    TensorGraph::TensorNode *s = graph.emplace_data({2});
    s->set_name("s");
    gt::unregister(s);
    std::size_t n = gt::append_invalidates_for_unmarked_unsealed(graph);
    REQUIRE(n == 0);
    REQUIRE(graph.ops().size() == 1);
    REQUIRE(graph.ops().back()->op_name() == "UNREGISTER");
    REQUIRE(graph.ops().back()->inputs()[0] == s);
}

TEST_CASE("TensorRef last-drop records UNREGISTER", "[graph][tensor]")
{
    TensorGraph graph("test");
    TensorGraph::TensorNode *raw = nullptr;
    {
        nntile::TensorRef x = graph.data({2});
        raw = x.get();
    }
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "UNREGISTER");
    REQUIRE(graph.ops()[0]->inputs()[0] == raw);
}
