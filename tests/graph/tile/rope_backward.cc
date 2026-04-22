/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/rope_backward.cc
 * Test TileGraph rope backward vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/rope_backward.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/rope_backward.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph rope_backward", "[graph][tile]")
{
    const std::vector<Index> sh = {2}, tsh = {4,5};
    TileGraph g("g");
    auto* si = g.data(sh, "si", DataType::FP32);
    auto* co = g.data(sh, "co", DataType::FP32);
    auto* dy = g.data(tsh, "dy", DataType::FP32);
    auto* dx = g.data(tsh, "dx", DataType::FP32);
    si->mark_input(true); co->mark_input(true); dy->mark_input(true);
    dx->mark_input(true); dx->mark_output(true);
    tg::rope_backward(si, co, dy, dx);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> siv(2), cev(2), dyy(20), dxx(20, 0.01f);
    for(int i=0;i<2;++i){siv[static_cast<size_t>(i)]=0.1f; cev[static_cast<size_t>(i)]=0.2f;}
    for(int i=0;i<20;++i) dyy[static_cast<size_t>(i)]=0.05f*static_cast<float>(i+1);
    r.bind_data("si", siv); r.bind_data("co", cev); r.bind_data("dy", dyy); r.bind_data("dx", dxx);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("dx");
    nntile::tile::Tile<fp32_t> Si(sh), Co(sh), DY(tsh), DX(tsh);
    using Y = typename fp32_t::repr_t;
    { auto a=Si.acquire(STARPU_W),b=Co.acquire(STARPU_W);
      for(int i=0;i<2;++i){a[i]=Y(0.1f);b[i]=Y(0.2f);} a.release(); b.release(); }
    { auto c=DY.acquire(STARPU_W);
      for(int i=0;i<20;++i) c[i]=Y(0.05f*static_cast<float>(i+1)); c.release(); }
    { auto d=DX.acquire(STARPU_W);
      for(int i=0;i<20;++i) d[i]=Y(0.01f);
      d.release(); }
    nntile::tile::rope_backward<fp32_t>(Si, Co, DY, DX);
    starpu_task_wait_for_all();
    std::vector<float> tr(20);
    { auto L=DX.acquire(STARPU_R);
      for(int i=0;i<20;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(int i=0;i<20;++i) REQUIRE(std::abs(gout[static_cast<size_t>(i)]-tr[static_cast<size_t>(i)])<1e+2f);
}