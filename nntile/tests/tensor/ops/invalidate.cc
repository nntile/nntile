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
    auto *x = graph.data({4, 5})->set_name("x");
    gt::invalidate(x);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "INVALIDATE");
    REQUIRE(graph.ops()[0]->inputs().size() == 1);
    REQUIRE(graph.ops()[0]->inputs()[0] == x);
    REQUIRE(graph.ops()[0]->outputs().empty());
}

TEST_CASE(
    "append_invalidates_for_unmarked_unsealed skips marked",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *keep = graph.data({2})->set_name("keep");
    auto *drop = graph.data({2})->set_name("drop");
    keep->mark_output(true);
    gt::clear(keep);
    gt::clear(drop);
    // drop is unmarked; keep is mark_output.
    std::size_t n = gt::append_invalidates_for_unmarked_unsealed(graph);
    REQUIRE(n == 1);
    REQUIRE(graph.ops().back()->op_name() == "INVALIDATE");
    REQUIRE(graph.ops().back()->inputs()[0] == drop);
}
