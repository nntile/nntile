/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/conv2d_bwd_input_inplace.cc
 * Test TileGraph conv2d bwd input inplace vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/conv2d_bwd_input_inplace.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/conv2d_bwd_input_inplace.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph conv2d_bwd_input_inplace", "[graph][tile]")
{
    const std::vector<Index> dYh={2,2,1,1}, Ch={2,2,1,1}, dXh={3,3,1,1};
    const Index n=4, nc=4, nx=9;
    TileGraph g("g");
    auto* dY = g.data(dYh, "dY", DataType::FP32);
    auto* Cg = g.data(Ch, "C", DataType::FP32);
    auto* dX = g.data(dXh, "dX", DataType::FP32);
    dY->mark_input(true); Cg->mark_input(true); dX->mark_input(true); dX->mark_output(true);
    tg::conv2d_bwd_input_inplace(2,2,1,1,1,1,2,2,1,1,1,0,0,1.0,dY,Cg,3,3,0.0,dX);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> a(n), b(nc), c(nx,0.f);
    for(Index i=0;i<4;++i) a[static_cast<size_t>(i)]=static_cast<float>(i+1);
    for(Index i=0;i<4;++i) b[static_cast<size_t>(i)]=static_cast<float>(i+1);
    r.bind_data("dY", a); r.bind_data("C", b); r.bind_data("dX", c);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("dX");
    nntile::tile::Tile<fp32_t> DY(dYh), CgT(Ch), DX(dXh);
    using Yf = typename fp32_t::repr_t;
    { auto p=DY.acquire(STARPU_W),q=CgT.acquire(STARPU_W),s=DX.acquire(STARPU_W);
      for(Index i=0;i<4;++i) {p[i]=Yf(static_cast<float>(i+1));q[i]=Yf(static_cast<float>(i+1));}
      for(Index i=0;i<9;++i) s[i]=Yf(0.0f);
      p.release();q.release();s.release(); }
    nntile::tile::conv2d_bwd_input_inplace<fp32_t>(2,2,1,1,1,1,2,2,1,1,1,0,0,1.0,DY,CgT,3,3,0.0,DX);
    starpu_task_wait_for_all();
    std::vector<float> tr(9);
    { auto L=DX.acquire(STARPU_R);
      for(Index i=0;i<9;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(Index i=0;i<9;++i) REQUIRE(std::abs(gout[static_cast<size_t>(i)]-tr[static_cast<size_t>(i)])<1e+2f);
}
