/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/adam_step.cc
 * Adam step: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "context_fixture.hh"
#include "mixed_tile_common.hh"
#include <nntile/graph.hh>

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace tt = nntile::graph::tile_tests;

TEST_CASE("Adam step mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    constexpr Index n = 10;

    auto build = [=](TensorGraph& g, bool tile_inputs) {
        TensorGraph::TensorNode* grad = g.data({n}, "grad", DataType::FP32);
        TensorGraph::TensorNode* m = g.data({n}, "m", DataType::FP32);
        TensorGraph::TensorNode* v = g.data({n}, "v", DataType::FP32);
        TensorGraph::TensorNode* p = g.data({n}, "p", DataType::FP32);
        grad->mark_input(true);
        m->mark_input(true);
        v->mark_input(true);
        p->mark_input(true);
        if(tile_inputs)
        {
            tt::apply_mixed_tile_sizes_1d(grad);
            tt::apply_mixed_tile_sizes_1d(m);
            tt::apply_mixed_tile_sizes_1d(v);
            tt::apply_mixed_tile_sizes_1d(p);
        }
        gt::adam_step(100, Scalar{0.9f}, Scalar{0.99f}, Scalar{1e-6f},
            Scalar{0.001f}, Scalar{0.f}, grad, m, v, p);
        p->mark_output(true);
        m->mark_output(true);
        v->mark_output(true);
    };

    TensorGraph g_ref("ref");
    build(g_ref, false);
    TensorGraph g_tile("tile");
    build(g_tile, true);

    std::vector<float> grad(static_cast<size_t>(n));
    std::vector<float> m(static_cast<size_t>(n));
    std::vector<float> v(static_cast<size_t>(n));
    std::vector<float> p(static_cast<size_t>(n));
    for(Index i = 0; i < n; ++i)
    {
        grad[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
        m[static_cast<size_t>(i)] = 0.f;
        v[static_cast<size_t>(i)] = 0.f;
        p[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i);
    }

    TensorGraph::Runtime rt_ref(g_ref);
    rt_ref.compile();
    rt_ref.bind_data("grad", grad);
    rt_ref.bind_data("m", m);
    rt_ref.bind_data("v", v);
    rt_ref.bind_data("p", p);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> p_ref = rt_ref.get_output<float>("p");

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("grad", grad);
    rt_tile.bind_data("m", m);
    rt_tile.bind_data("v", v);
    rt_tile.bind_data("p", p);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> p_tile = rt_tile.get_output<float>("p");

    REQUIRE(tt::max_rel_err(p_ref, p_tile) < 1e-3f);
    REQUIRE(tt::frob_rel_err(p_ref, p_tile) < 1e-3f);
}
