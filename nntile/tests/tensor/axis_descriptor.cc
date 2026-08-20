/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/axis_descriptor.cc
 * Tests for AxisDescriptor and eager axis merging in TensorGraph.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("Fresh tensors have independent axis descriptors", "[graph][axis]")
{
    TensorGraph graph("fresh");
    nntile::TensorRef x = graph.data({5, 4});
    x->set_name("x");
    nntile::TensorRef y = graph.data({5, 4});
    y->set_name("y");

    REQUIRE(x->axis(0) != y->axis(0));
    REQUIRE(x->axis(1) != y->axis(1));

    REQUIRE(x->axis(0)->extent == 5);
    REQUIRE(x->axis(1)->extent == 4);
    REQUIRE(x->axis(0)->members.size() == 1);
}

TEST_CASE("add merges axis groups eagerly", "[graph][axis]")
{
    TensorGraph graph("add_merge");
    nntile::TensorRef x = graph.data({5, 4});
    x->set_name("x");
    nntile::TensorRef y = graph.data({5, 4});
    y->set_name("y");
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::add(1.0, x, 1.0, y));
    z->set_name("z");

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
    nntile::TensorRef x = graph.data({4, 3});
    x->set_name("x");
    nntile::TensorRef y = graph.data({4, 3});
    y->set_name("y");
    gt::add_inplace(1.0, x, 1.0, y);

    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(1) == y->axis(1));
    REQUIRE(x->axis(0)->members.size() == 2);
}

TEST_CASE("Axis merging is transitive through chains", "[graph][axis]")
{
    TensorGraph graph("chain");
    nntile::TensorRef a = graph.data({4});
    a->set_name("a");
    nntile::TensorRef b = graph.data({4});
    b->set_name("b");
    nntile::TensorRef c = nntile::TensorRef::adopt(gt::add(1.0, a, 1.0, b));
    c->set_name("c");

    nntile::TensorRef d = graph.data({4});

    c->set_name("d");
    nntile::TensorRef e = nntile::TensorRef::adopt(gt::add(1.0, c, 1.0, d));
    e->set_name("e");

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
    nntile::TensorRef x = graph.data({3, 2});
    x->set_name("x");
    nntile::TensorRef y = graph.data({3, 2});
    y->set_name("y");
    nntile::TensorRef w = nntile::TensorRef::adopt(gt::add(1.0, x, 1.0, y));
    w->set_name("w");
    nntile::TensorRef v = nntile::TensorRef::adopt(gt::add(1.0, w, 1.0, y));
    v->set_name("v");
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::add(1.0, v, 1.0, w));
    z->set_name("z");

    // All 5 tensors share same axis descriptors per dimension
    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(0) == w->axis(0));
    REQUIRE(x->axis(0) == v->axis(0));
    REQUIRE(x->axis(0) == z->axis(0));

    REQUIRE(x->axis(1) == z->axis(1));
    REQUIRE(x->axis(0)->members.size() == 5);
    REQUIRE(x->axis(1)->members.size() == 5);
}

TEST_CASE("Axis naming propagates through group")
{
    TensorGraph graph("naming");
    nntile::TensorRef x = graph.data({5, 4});
    x->set_name("x");
    nntile::TensorRef y = graph.data({5, 4});
    y->set_name("y");
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::add(1.0, x, 1.0, y));
    z->set_name("z");

    // Name from one tensor is visible from all
    x->axis(0)->name = "batch";
    REQUIRE(y->axis(0)->name == "batch");

    x->axis(1)->name = "features";
    REQUIRE(z->axis(1)->name == "features");
}

TEST_CASE("Axis merge rejects different extents")
{
    TensorGraph graph("mismatch");
    nntile::TensorRef x = graph.data({4});
    x->set_name("x");
    nntile::TensorRef y = graph.data({5});
    y->set_name("y");

    REQUIRE_THROWS_AS(gt::add(1.0, x, 1.0, y), std::invalid_argument);
}

TEST_CASE("set_axes shares axis groups with another tensor", "[graph][axis]")
{
    TensorGraph graph("shared_axes");
    nntile::TensorRef x = graph.data({5, 4});
    x->set_name("x");

    nntile::TensorRef y = graph.data({5, 4});
    y->set_name("y");
    y->set_axes(x->axes());
    REQUIRE(x->axis(0) == y->axis(0));
    REQUIRE(x->axis(1) == y->axis(1));
    REQUIRE(x->axis(0)->members.size() == 2);
    REQUIRE(y->shape() == x->shape());
}

TEST_CASE("Axis merge preserves name from replaced group", "[graph][axis]")
{
    TensorGraph graph("name_preserve");
    nntile::TensorRef x = graph.data({4});
    x->set_name("x");
    nntile::TensorRef y = graph.data({4});
    y->set_name("y");

    y->axis(0)->name = "my_axis";
    gt::add_inplace(1.0, x, 1.0, y);

    // The name from y's axis should be preserved in the merged group
    REQUIRE(x->axis(0)->name == "my_axis");
}

TEST_CASE("merge_axis unions by size into the larger group", "[graph][axis]")
{
    // Capture cost: merging a fresh 1-member axis into a large group must not
    // walk every historical member (union-by-size keeps the large descriptor).
    TensorGraph graph("union_by_size");
    nntile::TensorRef hub = graph.data({8});
    hub->set_name("hub");
    for (int i = 0; i < 16; ++i)
    {
        nntile::TensorRef leaf = graph.data({8});
        // Prefer the large group as the first arg once it outgrows the leaf.
        merge_axis(hub->mutable_axes()[0], leaf->mutable_axes()[0]);
        REQUIRE(leaf->axis(0) == hub->axis(0));
    }
    AxisDescriptor *hub_axis = hub->axis(0);
    REQUIRE(hub_axis->members.size() == 17);

    nntile::TensorRef fresh = graph.data({8});
    fresh->set_name("fresh");
    // First arg is the small side (as gemm often does for activations).
    merge_axis(fresh->mutable_axes()[0], hub->mutable_axes()[0]);
    REQUIRE(fresh->axis(0) == hub_axis);
    REQUIRE(hub->axis(0) == hub_axis);
    REQUIRE(hub_axis->members.size() == 18);
}

TEST_CASE("Self-add (x == y) is rejected", "[graph][axis]")
{
    TensorGraph graph("self_add");
    nntile::TensorRef x = graph.data({4, 3});
    x->set_name("x");

    REQUIRE_THROWS_AS(gt::add(2.0, x, 3.0, x), std::invalid_argument);
}

TEST_CASE("Dead TensorNode IR is GC'd and leaves holes", "[graph][axis][gc]")
{
    TensorGraph graph("gc_holes");
    TensorGraph::TensorNode *a = graph.emplace_data({4});
    TensorGraph::TensorNode *b = graph.emplace_data({4});
    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_live_data() == 2);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    std::vector<TensorGraph::TensorNode *> dead =
        graph.collect_dead_data_nodes();
    REQUIRE(dead.size() == 2);
    graph.destroy_data_nodes(dead);
    REQUIRE(graph.num_live_data() == 0);
    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.axis_groups().empty());
}

TEST_CASE("GC keeps live TensorRefs and prunes axis members",
    "[graph][axis][gc]")
{
    TensorGraph graph("gc_members");
    nntile::TensorRef live = graph.data({8});
    live->set_name("live");
    {
        nntile::TensorRef tmp = graph.data({8});
        tmp->set_name("tmp");
        merge_axis(live->mutable_axes()[0], tmp->mutable_axes()[0]);
        REQUIRE(live->axis(0)->members.size() == 2);
        REQUIRE(graph.axis_groups().size() == 1);
        REQUIRE(graph.collect_dead_data_nodes().empty());
    }
    // Last TensorRef drop records UNREGISTER; the node stays referenced
    // until those ops are sealed and dropped.
    REQUIRE(graph.num_ops() >= 1);
    REQUIRE(graph.collect_dead_data_nodes().empty());
    graph.seal_phase();
    graph.drop_all_ops();
    std::vector<TensorGraph::TensorNode *> dead =
        graph.collect_dead_data_nodes();
    REQUIRE(dead.size() == 1);
    graph.destroy_data_nodes(dead);
    REQUIRE(graph.num_live_data() == 1);
    REQUIRE(graph.num_data() == 2);
    REQUIRE(live->axis(0)->members.size() == 1);
    REQUIRE(graph.axis_groups().size() == 1);
}