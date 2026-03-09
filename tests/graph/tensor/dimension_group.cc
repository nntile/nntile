/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/dimension_group.cc
 * Tests for automatic dimension group discovery on TensorGraph.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/dimension_group.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

const DimensionGroup* find_group_containing(
    const std::vector<DimensionGroup>& groups,
    const std::string& tensor_name, int axis)
{
    for(const auto& g : groups)
    {
        for(const auto& m : g.members)
        {
            if(m.node->name() == tensor_name && m.axis == axis)
                return &g;
        }
    }
    return nullptr;
}

bool same_group(const std::vector<DimensionGroup>& groups,
                const std::string& t1, int a1,
                const std::string& t2, int a2)
{
    const auto* g1 = find_group_containing(groups, t1, a1);
    const auto* g2 = find_group_containing(groups, t2, a2);
    return g1 != nullptr && g1 == g2;
}

} // namespace

TEST_CASE("Dimension groups: single tensor, no ops", "[graph][dimgroup]")
{
    TensorGraph graph("single");
    graph.data({4, 5}, "x");

    auto groups = discover_dimension_groups(graph);
    REQUIRE(groups.size() == 2);

    auto* g0 = find_group_containing(groups, "x", 0);
    auto* g1 = find_group_containing(groups, "x", 1);
    REQUIRE(g0 != nullptr);
    REQUIRE(g1 != nullptr);
    REQUIRE(g0 != g1);
    REQUIRE(g0->extent == 4);
    REQUIRE(g1->extent == 5);
    REQUIRE(g0->members.size() == 1);
    REQUIRE(g1->members.size() == 1);
}

TEST_CASE("Dimension groups: elementwise add", "[graph][dimgroup]")
{
    TensorGraph graph("ew_add");
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");
    gt::add(1.0, x, 1.0, y, "z");

    auto groups = discover_dimension_groups(graph);

    // Two groups: dim0 group and dim1 group
    // Each group contains x, y, z for that dimension
    REQUIRE(same_group(groups, "x", 0, "y", 0));
    REQUIRE(same_group(groups, "x", 0, "z", 0));
    REQUIRE(same_group(groups, "x", 1, "y", 1));
    REQUIRE(same_group(groups, "x", 1, "z", 1));

    // dim0 and dim1 are in different groups
    REQUIRE_FALSE(same_group(groups, "x", 0, "x", 1));

    auto* g0 = find_group_containing(groups, "x", 0);
    REQUIRE(g0->members.size() == 3);
    REQUIRE(g0->extent == 4);
}

TEST_CASE("Dimension groups: diamond add", "[graph][dimgroup]")
{
    TensorGraph graph("diamond");
    auto* x = graph.data({2, 3}, "x");
    auto* y = graph.data({2, 3}, "y");
    auto* w = gt::add(1.0, x, 1.0, y, "w");
    auto* v = gt::add(1.0, w, 1.0, y, "v");
    gt::add(1.0, v, 1.0, w, "z");

    auto groups = discover_dimension_groups(graph);

    // All 5 tensors share the same two groups
    REQUIRE(same_group(groups, "x", 0, "z", 0));
    REQUIRE(same_group(groups, "y", 0, "w", 0));
    REQUIRE(same_group(groups, "v", 0, "z", 0));

    REQUIRE(same_group(groups, "x", 1, "z", 1));
    REQUIRE(same_group(groups, "y", 1, "w", 1));

    auto* g0 = find_group_containing(groups, "x", 0);
    REQUIRE(g0->members.size() == 5);

    auto* g1 = find_group_containing(groups, "x", 1);
    REQUIRE(g1->members.size() == 5);
}

TEST_CASE("Dimension groups: gemm basic", "[graph][dimgroup]")
{
    // y = x @ w: x [M,K], w [K,N], y [M,N]
    TensorGraph graph("gemm");
    auto* x = graph.data({8, 4}, "x");
    auto* w = graph.data({4, 6}, "w");
    gt::gemm(x, w, "y", 1.0, false, false, 1, 0);

    auto groups = discover_dimension_groups(graph);

    // M group: x.0, y.0
    REQUIRE(same_group(groups, "x", 0, "y", 0));

    // K group: x.1, w.0
    REQUIRE(same_group(groups, "x", 1, "w", 0));

    // N group: w.1, y.1
    REQUIRE(same_group(groups, "w", 1, "y", 1));

    // All three groups are different
    REQUIRE_FALSE(same_group(groups, "x", 0, "x", 1));
    REQUIRE_FALSE(same_group(groups, "x", 0, "w", 1));
    REQUIRE_FALSE(same_group(groups, "x", 1, "w", 1));

    auto* gM = find_group_containing(groups, "x", 0);
    auto* gK = find_group_containing(groups, "x", 1);
    auto* gN = find_group_containing(groups, "w", 1);
    REQUIRE(gM->extent == 8);
    REQUIRE(gK->extent == 4);
    REQUIRE(gN->extent == 6);
}

TEST_CASE("Dimension groups: gemm transposed", "[graph][dimgroup]")
{
    // y = x^T @ w: x [K,M] (trans_a=true), w [K,N], y [M,N]
    TensorGraph graph("gemm_t");
    auto* x = graph.data({4, 8}, "x");
    auto* w = graph.data({4, 6}, "w");
    gt::gemm(x, w, "y", 1.0, true, false, 1, 0);

    auto groups = discover_dimension_groups(graph);

    // M: x.1, y.0
    REQUIRE(same_group(groups, "x", 1, "y", 0));
    // K: x.0, w.0
    REQUIRE(same_group(groups, "x", 0, "w", 0));
    // N: w.1, y.1
    REQUIRE(same_group(groups, "w", 1, "y", 1));
}

TEST_CASE("Dimension groups: gemm with batch", "[graph][dimgroup]")
{
    // y = x @ w batched: x [M,K,B], w [K,N,B], y [M,N,B]
    TensorGraph graph("gemm_batch");
    auto* x = graph.data({8, 4, 3}, "x");
    auto* w = graph.data({4, 6, 3}, "w");
    gt::gemm(x, w, "y", 1.0, false, false, 1, 1);

    auto groups = discover_dimension_groups(graph);

    REQUIRE(same_group(groups, "x", 0, "y", 0));   // M
    REQUIRE(same_group(groups, "x", 1, "w", 0));   // K
    REQUIRE(same_group(groups, "w", 1, "y", 1));   // N
    REQUIRE(same_group(groups, "x", 2, "y", 2));   // B
    REQUIRE(same_group(groups, "w", 2, "y", 2));   // B
}

TEST_CASE("Dimension groups: MLP pattern", "[graph][dimgroup]")
{
    // h = x @ w1, a = gelu(h), y = a @ w2
    // x [B,M], w1 [M,H], h [B,H], a [B,H], w2 [H,N], y [B,N]
    TensorGraph graph("mlp");
    auto* x = graph.data({16, 64}, "x");
    auto* w1 = graph.data({64, 128}, "w1");
    auto* h = gt::gemm(x, w1, "h", 1.0, false, false, 1, 0);

    auto* a = graph.data({16, 128}, "a");
    gt::gelu(h, a);

    auto* w2 = graph.data({128, 32}, "w2");
    gt::gemm(a, w2, "y", 1.0, false, false, 1, 0);

    auto groups = discover_dimension_groups(graph);

    // B group: x.0, h.0, a.0, y.0
    REQUIRE(same_group(groups, "x", 0, "h", 0));
    REQUIRE(same_group(groups, "h", 0, "a", 0));
    REQUIRE(same_group(groups, "a", 0, "y", 0));

    // M group: x.1, w1.0
    REQUIRE(same_group(groups, "x", 1, "w1", 0));

    // H group: w1.1, h.1, a.1, w2.0
    REQUIRE(same_group(groups, "w1", 1, "h", 1));
    REQUIRE(same_group(groups, "h", 1, "a", 1));
    REQUIRE(same_group(groups, "a", 1, "w2", 0));

    // N group: w2.1, y.1
    REQUIRE(same_group(groups, "w2", 1, "y", 1));

    // All four groups are distinct
    REQUIRE_FALSE(same_group(groups, "x", 0, "x", 1));
    REQUIRE_FALSE(same_group(groups, "x", 0, "w1", 1));
    REQUIRE_FALSE(same_group(groups, "x", 1, "w2", 1));
}

TEST_CASE("Dimension groups: add_inplace chain", "[graph][dimgroup]")
{
    TensorGraph graph("inplace");
    auto* x = graph.data({3, 4}, "x");
    auto* y = graph.data({3, 4}, "y");
    auto* z = graph.data({3, 4}, "z");
    gt::add_inplace(1.0, x, 1.0, y);
    gt::add_inplace(1.0, y, 1.0, z);

    auto groups = discover_dimension_groups(graph);

    REQUIRE(same_group(groups, "x", 0, "y", 0));
    REQUIRE(same_group(groups, "y", 0, "z", 0));
    REQUIRE(same_group(groups, "x", 1, "z", 1));
}

TEST_CASE("Dimension groups: transpose", "[graph][dimgroup]")
{
    // transpose with ndim=1: cyclic rotation by 1
    // src [A, B, C] -> dst [B, C, A]
    TensorGraph graph("transpose");
    auto* x = graph.data({2, 3, 5}, "x");
    gt::transpose(1.0, x, "y", 1);

    auto groups = discover_dimension_groups(graph);

    // src.dim[(i + ndim) % n] == dst.dim[i]
    // ndim=1, n=3: src.1->dst.0, src.2->dst.1, src.0->dst.2
    REQUIRE(same_group(groups, "x", 1, "y", 0));
    REQUIRE(same_group(groups, "x", 2, "y", 1));
    REQUIRE(same_group(groups, "x", 0, "y", 2));
}

TEST_CASE("Dimension groups: sum_slice reduces axis", "[graph][dimgroup]")
{
    // src [4, 5, 6] -> sum_slice(axis=1) -> dst [4, 6]
    TensorGraph graph("sum_slice");
    auto* src = graph.data({4, 5, 6}, "src");
    gt::sum_slice(src, "dst", 1, 0, 1.0, 0.0);

    auto groups = discover_dimension_groups(graph);

    // src.0 == dst.0, src.2 == dst.1, src.1 is standalone
    REQUIRE(same_group(groups, "src", 0, "dst", 0));
    REQUIRE(same_group(groups, "src", 2, "dst", 1));
    REQUIRE_FALSE(same_group(groups, "src", 1, "dst", 0));
    REQUIRE_FALSE(same_group(groups, "src", 1, "dst", 1));
}

TEST_CASE("Dimension groups: fill and clear have no constraints",
          "[graph][dimgroup]")
{
    TensorGraph graph("fill_clear");
    auto* x = graph.data({2, 3}, "x");
    auto* y = graph.data({2, 3}, "y");
    gt::fill(1.0, x);
    gt::clear(y);

    auto groups = discover_dimension_groups(graph);

    // x and y are independent (no op connects them)
    REQUIRE_FALSE(same_group(groups, "x", 0, "y", 0));
    REQUIRE_FALSE(same_group(groups, "x", 1, "y", 1));
    REQUIRE(groups.size() == 4);
}

TEST_CASE("Dimension groups: group naming from members", "[graph][dimgroup]")
{
    TensorGraph graph("naming");
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");
    gt::add(1.0, x, 1.0, y, "z");

    auto groups = discover_dimension_groups(graph);

    // Groups are auto-named from first member (sorted by tensor name, axis)
    for(const auto& g : groups)
    {
        REQUIRE_FALSE(g.name.empty());
        REQUIRE(g.members.size() == 3);
    }
}

TEST_CASE("Dimension groups: scale (elementwise same-shape)",
          "[graph][dimgroup]")
{
    TensorGraph graph("scale");
    auto* x = graph.data({3, 4}, "x");
    gt::scale(2.0, x, "y");

    auto groups = discover_dimension_groups(graph);

    REQUIRE(same_group(groups, "x", 0, "y", 0));
    REQUIRE(same_group(groups, "x", 1, "y", 1));
}
