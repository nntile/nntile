/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/rope.cc
 * Test TileGraph rope vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/rope.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/rope.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph rope", "[graph][tile]")
{
    const std::vector<Index> sh = {2}, tsh = {4,5};
    const Index n2=2, n=20;
    TileGraph g("g");
    auto* si = g.data(sh, "si", DataType::FP32);
    auto* co = g.data(sh, "co", DataType::FP32);
    auto* sr = g.data(tsh, "src", DataType::FP32);
    auto* d = g.data(tsh, "d", DataType::FP32);
    si->mark_input(true); co->mark_input(true); sr->mark_input(true); d->mark_output(true);
    tg::rope(si, co, sr, d);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> sv(2), cv(2), src(20,0.03f);
    for(int i=0;i<2;++i) { sv[static_cast<size_t>(i)]=0.1f*static_cast<float>(i+1);
      cv[static_cast<size_t>(i)]=0.2f*static_cast<float>(i+1);}
    for(int i=0;i<20;++i) src[static_cast<size_t>(i)]=0.03f*static_cast<float>(i+1);
    std::vector<float> dv(20,0.f);
    r.bind_data("si", sv);
    r.bind_data("co", cv);
    r.bind_data("src", src);
    r.bind_data("d", dv);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<fp32_t> Si(sh), Co(sh), Src(tsh), D(tsh);
    using Y = typename fp32_t::repr_t;
    { auto a=Si.acquire(STARPU_W),b=Co.acquire(STARPU_W);
      for(int i=0;i<2;++i) {a[i]=Y(sv[static_cast<size_t>(i)]); b[i]=Y(cv[static_cast<size_t>(i)]);}a.release();b.release(); }
    { auto c=Src.acquire(STARPU_W);
      for(int i=0;i<20;++i) c[i]=Y(src[static_cast<size_t>(i)]); c.release(); }
    { auto d0=D.acquire(STARPU_W); for(int i=0;i<20;++i) d0[i]=Y(0.0f); d0.release(); }
    nntile::tile::rope<fp32_t>(Si, Co, Src, D);
    starpu_task_wait_for_all();
    std::vector<float> tr(20);
    { auto L=D.acquire(STARPU_R);
      for(int i=0;i<20;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(int i=0;i<20;++i) REQUIRE(std::abs(gout[static_cast<size_t>(i)]-tr[static_cast<size_t>(i)])<1e+2f);
}
