/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/total_sum_accum.cc
 * Test TileGraph total sum accum vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/total_sum_accum.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/total_sum_accum.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph total_sum_accum", "[graph][tile]")
{
    const std::vector<Index> leh = {2,2}, srh = {3,2,2}, clh = {2,2}, vh = std::vector<Index>{};
    const Scalar a = 1.0; const Index ign = -1;
    TileGraph g("g");
    auto* l = g.data(leh, "lse", DataType::FP32);
    auto* s = g.data(srh, "src", DataType::FP32);
    auto* c = g.data(clh, "cl", DataType::INT64);
    auto* v = g.data(vh, "val", DataType::FP32);
    l->mark_input(true); s->mark_input(true); c->mark_input(true); v->mark_input(true); v->mark_output(true);
    tg::total_sum_accum(a, l, s, c, v, ign);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> lse(4), src(3*2*2), v0(1,0.f);
    std::vector<std::int64_t> cl(4);
    for(Index i=0;i<4;++i) lse[static_cast<size_t>(i)]=0.1f*static_cast<float>(i+1);
    for(Index i=0;i<3*2*2;++i) src[static_cast<size_t>(i)]=0.05f*static_cast<float>(i+1);
    cl[0]=0; cl[1]=1; cl[2]=2; cl[3]=0;
    r.bind_data("lse", lse);
    r.bind_data("src", src);
    r.bind_data("cl", cl);
    r.bind_data("val", v0);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>("val");
    nntile::tile::Tile<fp32_t> L(leh), S(srh), Vref(vh);
    nntile::tile::Tile<nntile::int64_t> C(clh);
    using Y = typename fp32_t::repr_t;
    { auto a1=L.acquire(STARPU_W),a2=S.acquire(STARPU_W);
      for(Index i=0;i<4;++i) a1[i]=Y(lse[static_cast<size_t>(i)]);
      for(Index i=0;i<3*2*2;++i) a2[i]=Y(src[static_cast<size_t>(i)]);
      a1.release(); a2.release(); }
    { auto b=C.acquire(STARPU_W);
      b[0]=0; b[1]=1; b[2]=2; b[3]=0; b.release(); }
    { auto z=Vref.acquire(STARPU_W); z[0]=Y(0.0f); z.release(); }
    nntile::tile::total_sum_accum<fp32_t>(a, L, S, C, Vref, ign);
    starpu_task_wait_for_all();
    float tref=0; { auto L2=Vref.acquire(STARPU_R); tref=static_cast<float>(L2[0]); L2.release(); }
    REQUIRE(std::abs(gout[0]-tref)<1e-3f);
}
