/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tensor/ops/invalidate.cc
 * Test TensorGraph invalidate structure and unmarked-phase helper.
 */

#include "nntile/tensor/ops/invalidate.hh"

#include "nntile/tensor.hh"
#include "nntile/tensor/ops/clear.hh"

#include <catch2/catch_test_macros.hpp>

using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph invalidate structure", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef x = graph.data({4, 5});
    x->set_name("x");
    gt::invalidate(x);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "INVALIDATE");
    REQUIRE(graph.ops()[0]->inputs().size() == 1);
    REQUIRE(graph.ops()[0]->inputs()[0] == x);
    REQUIRE(graph.ops()[0]->outputs().empty());
}

TEST_CASE(
    "append_invalidates_for_unmarked_unsealed skips live TensorRef",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef keep = graph.data({2});
    keep->set_name("keep");
    TensorGraph::TensorNode *drop_raw = nullptr;
    {
        nntile::TensorRef drop = graph.data({2});
        drop->set_name("drop");
        drop_raw = drop.get();
        gt::clear(keep);
        gt::clear(drop);
    }
    std::size_t n = gt::append_invalidates_for_unmarked_unsealed(graph);
    REQUIRE(n == 1);
    REQUIRE(graph.ops().back()->op_name() == "INVALIDATE");
    REQUIRE(graph.ops().back()->inputs()[0] == drop_raw);
}
