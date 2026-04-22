/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/sgd_step.cc
 * SGD step: TensorGraph vs TileGraph (mixed tile sizes) parity.
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

TEST_CASE("SGD step mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    auto build = [](TensorGraph& g, bool tile_inputs) {
        TensorGraph::TensorNode* grad =
            g.data({10, 12}, "grad", DataType::FP32);
        TensorGraph::TensorNode* vel =
            g.data({10, 12}, "vel", DataType::FP32);
        TensorGraph::TensorNode* p = g.data({10, 12}, "p", DataType::FP32);
        grad->mark_input(true);
        vel->mark_input(true);
        p->mark_input(true);
        if(tile_inputs)
        {
            tt::apply_mixed_tile_sizes_2d(grad);
            tt::apply_mixed_tile_sizes_2d(vel);
            tt::apply_mixed_tile_sizes_2d(p);
        }
        gt::sgd_step(0, Scalar{0.9f}, Scalar{0.05f}, Scalar{0.f},
            Scalar{0.f}, false, grad, vel, p);
        p->mark_output(true);
        vel->mark_output(true);
    };

    TensorGraph g_ref("ref");
    build(g_ref, false);
    TensorGraph g_tile("tile");
    build(g_tile, true);

    std::vector<float> grad(10 * 12), vel(10 * 12), p(10 * 12);
    for(size_t i = 0; i < grad.size(); ++i)
    {
        grad[i] = 0.01f * static_cast<float>(static_cast<int>(i % 5) - 2);
        vel[i] = 0.f;
        p[i] = 0.2f;
    }

    TensorGraph::Runtime rt_ref(g_ref);
    rt_ref.compile();
    rt_ref.bind_data("grad", grad);
    rt_ref.bind_data("vel", vel);
    rt_ref.bind_data("p", p);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> p_ref = rt_ref.get_output<float>("p");

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("grad", grad);
    rt_tile.bind_data("vel", vel);
    rt_tile.bind_data("p", p);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> p_tile = rt_tile.get_output<float>("p");

    REQUIRE(tt::max_rel_err(p_ref, p_tile) < 1e-3f);
    REQUIRE(tt::frob_rel_err(p_ref, p_tile) < 1e-3f);
}
