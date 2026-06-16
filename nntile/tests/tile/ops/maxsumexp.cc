/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile_graph/maxsumexp.cc
 * Test TileGraph maxsumexp vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/maxsumexp.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/maxsumexp.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph maxsumexp axis0", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {3, 4, 5};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const std::vector<Index> stor_dh = {2, 4, 5};
    const std::vector<Index> graph_dh = graph_shape(stor_dh);
    const Index n1 = 60, n2 = 2 * 4 * 5;
    const Index stor_axis = 0;
    const Index g_axis =
        graph_axis(stor_axis, static_cast<Index>(stor_sh.size()));
    const int redux = 0;
    TileGraph g("g");
    auto* s = g.data(graph_sh, "s", DataType::FP32);
    auto* d = g.data(graph_dh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::maxsumexp(s, d, g_axis, redux);
    Runtime r(g);
    r.compile();
    std::vector<float> a(n1);
    std::vector<float> b(n2, 0.f);
    for(Index i = 0; i < n1; ++i)
    {
        a[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    r.bind_data(s, a);
    r.bind_data(d, b);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<fp32_t> S(stor_sh), D(stor_dh);
    using Y = typename fp32_t::repr_t;
    {
        auto p = S.acquire(STARPU_W);
        auto q = D.acquire(STARPU_W);
        for(Index i = 0; i < n1; ++i)
        {
            p[i] = Y(a[static_cast<size_t>(i)]);
        }
        for(Index j = 0; j < n2; ++j)
        {
            q[j] = Y(0);
        }
        p.release();
        q.release();
    }
    nntile::core::maxsumexp<fp32_t>(-1, S, D, stor_axis, redux);
    starpu_task_wait_for_all();
    std::vector<float> tr(n2);
    {
        auto L = D.acquire(STARPU_R);
        for(Index j = 0; j < n2; ++j)
        {
            tr[static_cast<size_t>(j)] = static_cast<float>(L[j]);
        }
        L.release();
    }
    for(size_t j = 0; j < tr.size(); ++j)
    {
        REQUIRE(std::abs(gout[j] - tr[j]) < 1e-6f);
    }
}
