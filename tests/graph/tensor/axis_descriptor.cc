/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/axis_descriptor.cc
 * Tests for AxisDescriptor and eager axis merging in TensorGraph.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE("Fresh tensors have independent axis descriptors",
          "[graph][axis]")
{
    TensorGraph graph("fresh");
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");

    REQUIRE(x->axis(0) != y->axis(0));
    REQUIRE(x->axis(1) != y->axis(1));

    REQUIRE(x->axis(0)->extent == 4);
    REQUIRE(x->axis(1)->extent == 5);
    REQUIRE(x->axis(0)->members.size() == 1);
}

TEST_CASE("add merges axis groups eagerly", "[graph][axis]")
{
    TensorGraph graph("add_merge");
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");
    auto* z = gt::add(1.0, x, 1.0, y, "z");

    // After add, all three tensors share the same axis descriptors
    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(0) == z->axis(0));
    REQUIRE(x->axis(1) == y->axis(1));
    REQUIRE(x->axis(1) == z->axis(1));

    // dim0 and dim1 are still different groups
    REQUIRE(x->axis(0) != x->axis(1));

    // Members list has all three tensors
    REQUIRE(x->axis(0)->members.size() == 3);
    REQUIRE(x->axis(1)->members.size() == 3);
}

TEST_CASE("add_inplace merges axis groups", "[graph][axis]")
{
    TensorGraph graph("inplace_merge");
    auto* x = graph.data({3, 4}, "x");
    auto* y = graph.data({3, 4}, "y");
    gt::add_inplace(1.0, x, 1.0, y);

    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(1) == y->axis(1));
    REQUIRE(x->axis(0)->members.size() == 2);
}

TEST_CASE("Axis merging is transitive through chains", "[graph][axis]")
{
    TensorGraph graph("chain");
    auto* a = graph.data({4}, "a");
    auto* b = graph.data({4}, "b");
    auto* c = gt::add(1.0, a, 1.0, b, "c");

    auto* d = graph.data({4}, "d");
    auto* e = gt::add(1.0, c, 1.0, d, "e");

    // a, b, c were merged in first add
    // c, d, e were merged in second add
    // So a, b, c, d, e should all share the same axis descriptor
    REQUIRE(a->axis(0) == b->axis(0));
    REQUIRE(a->axis(0) == c->axis(0));
    REQUIRE(a->axis(0) == d->axis(0));
    REQUIRE(a->axis(0) == e->axis(0));

    REQUIRE(a->axis(0)->members.size() == 5);
}

TEST_CASE("Axis merging is transitive: diamond pattern", "[graph][axis]")
{
    TensorGraph graph("diamond");
    auto* x = graph.data({2, 3}, "x");
    auto* y = graph.data({2, 3}, "y");
    auto* w = gt::add(1.0, x, 1.0, y, "w");
    auto* v = gt::add(1.0, w, 1.0, y, "v");
    auto* z = gt::add(1.0, v, 1.0, w, "z");

    // All 5 tensors share same axis descriptors per dimension
    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(0) == w->axis(0));
    REQUIRE(x->axis(0) == v->axis(0));
    REQUIRE(x->axis(0) == z->axis(0));

    REQUIRE(x->axis(1) == z->axis(1));
    REQUIRE(x->axis(0)->members.size() == 5);
    REQUIRE(x->axis(1)->members.size() == 5);
}

TEST_CASE("Axis naming propagates through group", "[graph][axis]")
{
    TensorGraph graph("naming");
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");
    gt::add(1.0, x, 1.0, y, "z");

    // Name from one tensor is visible from all
    x->axis(0)->name = "batch";
    REQUIRE(y->axis(0)->name == "batch");

    x->axis(1)->name = "features";
    auto* z = graph.get_tensor_node("z");
    REQUIRE(z->axis(1)->name == "features");
}

TEST_CASE("Axis merge rejects different extents", "[graph][axis]")
{
    TensorGraph graph("mismatch");
    auto* x = graph.data({4}, "x");
    auto* y = graph.data({5}, "y");

    REQUIRE_THROWS_AS(
        gt::add(1.0, x, 1.0, y, "z"), std::invalid_argument);
}

TEST_CASE("set_axes shares axis groups with another tensor",
          "[graph][axis]")
{
    TensorGraph graph("shared_axes");
    auto* x = graph.data({4, 5}, "x");

    auto* y = graph.data({4, 5}, "y");
    y->set_axes(x->axes());
    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(1) == y->axis(1));
    REQUIRE(x->axis(0)->members.size() == 2);
    REQUIRE(y->shape() == x->shape());
}

TEST_CASE("Axis merge preserves name from replaced group", "[graph][axis]")
{
    TensorGraph graph("name_preserve");
    auto* x = graph.data({4}, "x");
    auto* y = graph.data({4}, "y");

    y->axis(0)->name = "my_axis";
    gt::add_inplace(1.0, x, 1.0, y);

    // The name from y's axis should be preserved in the merged group
    REQUIRE(x->axis(0)->name == "my_axis");
}

TEST_CASE("Self-add (x == y) is rejected", "[graph][axis]")
{
    TensorGraph graph("self_add");
    auto* x = graph.data({3, 4}, "x");

    REQUIRE_THROWS_AS(gt::add(2.0, x, 3.0, x, "z"), std::invalid_argument);
}
