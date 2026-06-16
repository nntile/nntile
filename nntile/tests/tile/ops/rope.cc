/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile_graph/rope.cc
 * Test TileGraph rope vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/rope.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/rope.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph rope", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {2};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const std::vector<Index> stor_tsh = {4, 5};
    const std::vector<Index> graph_tsh = graph_shape(stor_tsh);
    const Index n = 20;
    TileGraph g("g");
    auto* si = g.data(graph_sh, "si", DataType::FP32);
    auto* co = g.data(graph_sh, "co", DataType::FP32);
    auto* sr = g.data(graph_tsh, "src", DataType::FP32);
    auto* d = g.data(graph_tsh, "d", DataType::FP32);
    si->mark_input(true);
    co->mark_input(true);
    sr->mark_input(true);
    d->mark_output(true);
    tg::rope(si, co, sr, d);
    Runtime r(g);
    r.compile();
    std::vector<float> sv(2), cv(2), src(n, 0.03f);
    for(int i = 0; i < 2; ++i)
    {
        sv[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        cv[static_cast<size_t>(i)] = 0.2f * static_cast<float>(i + 1);
    }
    for(int i = 0; i < 20; ++i)
    {
        src[static_cast<size_t>(i)] = 0.03f * static_cast<float>(i + 1);
    }
    std::vector<float> dv(n, 0.f);
    r.bind_data(si, sv);
    r.bind_data(co, cv);
    r.bind_data(sr, src);
    r.bind_data(d, dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<fp32_t> Si(stor_sh), Co(stor_sh), Src(stor_tsh), D(stor_tsh);
    using Y = typename fp32_t::repr_t;
    {
        auto a = Si.acquire(STARPU_W);
        auto b = Co.acquire(STARPU_W);
        for(int i = 0; i < 2; ++i)
        {
            a[i] = Y(sv[static_cast<size_t>(i)]);
            b[i] = Y(cv[static_cast<size_t>(i)]);
        }
        a.release();
        b.release();
    }
    {
        auto c = Src.acquire(STARPU_W);
        for(int i = 0; i < 20; ++i)
        {
            c[i] = Y(src[static_cast<size_t>(i)]);
        }
        c.release();
    }
    {
        auto d0 = D.acquire(STARPU_W);
        for(int i = 0; i < 20; ++i)
        {
            d0[i] = Y(0.0f);
        }
        d0.release();
    }
    nntile::core::rope<fp32_t>(-1, Si, Co, Src, D);
    starpu_task_wait_for_all();
    std::vector<float> tr(20);
    {
        auto L = D.acquire(STARPU_R);
        for(int i = 0; i < 20; ++i)
        {
            tr[static_cast<size_t>(i)] = static_cast<float>(L[i]);
        }
        L.release();
    }
    for(int i = 0; i < 20; ++i)
    {
        REQUIRE(std::abs(gout[static_cast<size_t>(i)] - tr[static_cast<size_t>(i)]) < 1e+2f);
    }
}
