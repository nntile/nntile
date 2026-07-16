/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/adam_step.cc
 * Adam step: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "mixed_tile_common.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/tile.hh>
#include <nntile/tensor.hh>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;
namespace tt = nntile::core_tests;

TEST_CASE("Adam step mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    constexpr Index n = 10;

    struct Nodes
    {
        nntile::TensorRef grad;
        nntile::TensorRef m;
        nntile::TensorRef v;
        nntile::TensorRef p;
    };

    auto build = [=](TensorGraph &g, bool tile_inputs) -> Nodes
    {
        Nodes nodes;
        nodes.grad = g.data({n}, DataType::FP32);
        nodes.grad->set_name("grad");
        nodes.m = g.data({n}, DataType::FP32);
        nodes.m->set_name("m");
        nodes.v = g.data({n}, DataType::FP32);
        nodes.v->set_name("v");
        nodes.p = g.data({n}, DataType::FP32);
        nodes.p->set_name("p");
        if (tile_inputs)
        {
            tt::apply_mixed_tile_sizes_1d(nodes.grad);
            tt::apply_mixed_tile_sizes_1d(nodes.m);
            tt::apply_mixed_tile_sizes_1d(nodes.v);
            tt::apply_mixed_tile_sizes_1d(nodes.p);
        }
        gt::adam_step(100,
            Scalar{0.9f},
            Scalar{0.99f},
            Scalar{1e-6f},
            Scalar{0.001f},
            Scalar{0.f},
            nodes.grad,
            nodes.m,
            nodes.v,
            nodes.p);
        return nodes;
    };

    TensorGraph g_ref("ref");
    Nodes ref_nodes = build(g_ref, false);
    TensorGraph g_tile("tile");
    Nodes tile_nodes = build(g_tile, true);

    std::vector<float> grad_h(static_cast<size_t>(n));
    std::vector<float> m_h(static_cast<size_t>(n));
    std::vector<float> v_h(static_cast<size_t>(n));
    std::vector<float> p_h(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
    {
        grad_h[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
        m_h[static_cast<size_t>(i)] = 0.f;
        v_h[static_cast<size_t>(i)] = 0.f;
        p_h[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i);
    }

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);

    Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data(ref_nodes.grad, grad_h);
    rt_ref.bind_data(ref_nodes.m, m_h);
    rt_ref.bind_data(ref_nodes.v, v_h);
    rt_ref.bind_data(ref_nodes.p, p_h);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> p_ref = rt_ref.get_output<float>(ref_nodes.p);

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data(tile_nodes.grad, grad_h);
    rt_tile.bind_data(tile_nodes.m, m_h);
    rt_tile.bind_data(tile_nodes.v, v_h);
    rt_tile.bind_data(tile_nodes.p, p_h);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> p_tile = rt_tile.get_output<float>(tile_nodes.p);

    nntile::test::require_relative_element_error(p_ref, p_tile);
}
