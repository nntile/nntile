/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/norm_fiber_inplace.cc
 * Test TileGraph norm fiber inplace vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/norm_fiber_inplace.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/norm_fiber_inplace.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph norm_fiber_inplace", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {5, 3, 20, 1};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const std::vector<Index> stor_dh = {5};
    const std::vector<Index> graph_dh = graph_shape(stor_dh);
    const Index n1 = 300, n2 = 5;
    const Scalar a = 1.0, b = 0.0;
    const Index stor_axis = 0, bd = 0;
    const Index g_axis =
        graph_axis(stor_axis, static_cast<Index>(stor_sh.size()));
    const int redux = 0;
    TileGraph g("g");
    auto* s = g.data(graph_sh, "s", DataType::FP32);
    auto* d = g.data(graph_dh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_input(true);
    d->mark_output(true);
    tg::norm_fiber_inplace(a, s, b, d, g_axis, bd, redux);
    Runtime rt(g);
    rt.compile();
    std::vector<float> v1(n1), v2(n2, -1.f);
    for(Index i = 0; i < n1; ++i) { v1[static_cast<size_t>(i)] = -1.0f; }
    rt.bind_data(s, v1);
    rt.bind_data(d, v2);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>(d);
    nntile::core::Tile<fp32_t> S(stor_sh), D(stor_dh);
    using Y = typename nntile::fp32_t::repr_t;
    { auto a1 = S.acquire(STARPU_W), a2 = D.acquire(STARPU_W);
      for(Index i = 0; i < n1; ++i) a1[i] = Y(-1.0f);
      for(Index j = 0; j < n2; ++j) a2[j] = Y(-1.0f);
      a1.release(); a2.release(); }
    nntile::core::norm_fiber_inplace<fp32_t>(-1, a, S, b, D, stor_axis, bd, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(5);
    { auto L = D.acquire(STARPU_R);
      for(Index j = 0; j < 5; ++j) tref[static_cast<size_t>(j)] = static_cast<float>(L[j]);
      L.release(); }
    for(int j = 0; j < 5; ++j) { REQUIRE(std::abs(gout[static_cast<size_t>(j)] - tref[static_cast<size_t>(j)]) < 1e-6f); }
}
