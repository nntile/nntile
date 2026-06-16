/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile_graph/softmax.cc
 * Test TileGraph softmax vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/softmax.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/softmax.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph softmax axis0", "[graph][tile]")
{
    const std::vector<Index> stor_mh = {2, 4, 5};
    const std::vector<Index> graph_mh = graph_shape(stor_mh);
    const std::vector<Index> stor_sh = {3, 4, 5};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const Index nms = 40, n = 60;
    const Scalar al = 1.0;
    const Index stor_axis = 0;
    const Index g_axis =
        graph_axis(stor_axis, static_cast<Index>(stor_sh.size()));
    TileGraph g("g");
    auto* m = g.data(graph_mh, "m", DataType::FP32);
    auto* s = g.data(graph_sh, "s", DataType::FP32);
    auto* d = g.data(graph_sh, "d", DataType::FP32);
    m->mark_input(true);
    s->mark_input(true);
    d->mark_output(true);
    tg::softmax(m, s, al, d, g_axis);
    Runtime r(g);
    r.compile();
    std::vector<float> mv(nms), sv(n), dv(n, 0.f);
    for(Index j = 0; j < nms; j += 2)
    {
        mv[static_cast<size_t>(j)] = static_cast<float>(j + 1);
        mv[static_cast<size_t>(j + 1)] =
            std::exp(static_cast<float>(j + 2) / 10.f);
    }
    for(Index i = 0; i < n; ++i)
    {
        sv[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
    }
    r.bind_data(m, mv);
    r.bind_data(s, sv);
    r.bind_data(d, dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<fp32_t> M(stor_mh), S(stor_sh), D(stor_sh);
    using Y = typename fp32_t::repr_t;
    {
        auto a = M.acquire(STARPU_W);
        auto b = S.acquire(STARPU_W);
        auto c = D.acquire(STARPU_W);
        for(Index j = 0; j < nms; j += 2)
        {
            a[j] = Y(mv[static_cast<size_t>(j)]);
            a[j + 1] = Y(mv[static_cast<size_t>(j + 1)]);
        }
        for(Index i = 0; i < n; ++i)
        {
            b[i] = Y(0.01f * static_cast<float>(i + 1));
            c[i] = Y(0);
        }
        a.release();
        b.release();
        c.release();
    }
    nntile::core::softmax<fp32_t>(-1, M, S, al, D, stor_axis);
    starpu_task_wait_for_all();
    std::vector<float> tr(n);
    {
        auto L = D.acquire(STARPU_R);
        for(Index i = 0; i < n; ++i)
        {
            tr[static_cast<size_t>(i)] = static_cast<float>(L[i]);
        }
        L.release();
    }
    for(size_t i = 0; i < tr.size(); ++i)
    {
        REQUIRE(std::abs(gout[i] - tr[i]) < 1e-6f);
    }
}
