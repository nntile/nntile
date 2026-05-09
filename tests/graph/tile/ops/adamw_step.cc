/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/adamw_step.cc
 * AdamW step: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "mixed_tile_common.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/graph.hh>
#include <vector>

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace tt = nntile::graph::tile_tests;

TEST_CASE("AdamW step mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    constexpr Index n = 10;

    auto build = [=](TensorGraph &g, bool tile_inputs)
    {
        TensorGraph::TensorNode *grad =
            g.data({n}, DataType::FP32)->set_name("grad");
        TensorGraph::TensorNode *m =
            g.data({n}, DataType::FP32)->set_name("m");
        TensorGraph::TensorNode *v =
            g.data({n}, DataType::FP32)->set_name("v");
        TensorGraph::TensorNode *p =
            g.data({n}, DataType::FP32)->set_name("p");
        grad->mark_input(true);
        m->mark_input(true);
        v->mark_input(true);
        p->mark_input(true);
        if (tile_inputs)
        {
            tt::apply_mixed_tile_sizes_1d(grad);
            tt::apply_mixed_tile_sizes_1d(m);
            tt::apply_mixed_tile_sizes_1d(v);
            tt::apply_mixed_tile_sizes_1d(p);
        }
        gt::adamw_step(100,
            Scalar{0.9f},
            Scalar{0.99f},
            Scalar{1e-6f},
            Scalar{0.001f},
            Scalar{0.01f},
            grad,
            m,
            v,
            p);
        p->mark_output(true);
    };

    TensorGraph g_ref("ref");
    build(g_ref, false);
    TensorGraph g_tile("tile");
    build(g_tile, true);

    std::vector<float> grad_h(static_cast<size_t>(n));
    std::vector<float> m_h(static_cast<size_t>(n));
    std::vector<float> v_h(static_cast<size_t>(n));
    std::vector<float> p_h(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
    {
        grad_h[static_cast<size_t>(i)] = 0.02f;
        m_h[static_cast<size_t>(i)] = 0.f;
        v_h[static_cast<size_t>(i)] = 0.f;
        p_h[static_cast<size_t>(i)] = 1.f;
    }

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);

    TileGraph::Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data(tt::tensor_node_named(g_ref, "grad"), grad_h);
    rt_ref.bind_data(tt::tensor_node_named(g_ref, "m"), m_h);
    rt_ref.bind_data(tt::tensor_node_named(g_ref, "v"), v_h);
    rt_ref.bind_data(tt::tensor_node_named(g_ref, "p"), p_h);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> p_ref =
        rt_ref.get_output<float>(tt::tensor_node_named(g_ref, "p"));

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data(tt::tensor_node_named(g_tile, "grad"), grad_h);
    rt_tile.bind_data(tt::tensor_node_named(g_tile, "m"), m_h);
    rt_tile.bind_data(tt::tensor_node_named(g_tile, "v"), v_h);
    rt_tile.bind_data(tt::tensor_node_named(g_tile, "p"), p_h);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> p_tile =
        rt_tile.get_output<float>(tt::tensor_node_named(g_tile, "p"));

    REQUIRE(tt::max_rel_err(p_ref, p_tile) < 1e-3f);
    REQUIRE(tt::frob_rel_err(p_ref, p_tile) < 1e-3f);
}
