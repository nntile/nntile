/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/conv2d_bwd_weight_inplace.cc
 * Test TileGraph conv2d bwd weight inplace vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/conv2d_bwd_weight_inplace.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/conv2d_bwd_weight_inplace.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph conv2d_bwd_weight_inplace", "[graph][tile]")
{
    const std::vector<Index> xh={3,3,1,1}, dyh={2,2,1,1}, dch={2,2,1,1};
    const Index nx=9, n4=4;
    TileGraph g("g");
    auto* X = g.data(xh, "X", DataType::FP32);
    auto* dY = g.data(dyh, "dY", DataType::FP32);
    auto* dC = g.data(dch, "dC", DataType::FP32);
    X->mark_input(true); dY->mark_input(true); dC->mark_input(true); dC->mark_output(true);
    tg::conv2d_bwd_weight_inplace(3,3,1,1,2,2,1,1,1,0,0,1.0,X,dY,2,2,1,1,0.0,dC);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> xv(nx), dyd(4), dcd(4,0.f);
    for(Index i=0;i<nx;++i) xv[static_cast<size_t>(i)]=static_cast<float>(i+1);
    for(Index i=0;i<4;++i) dyd[static_cast<size_t>(i)]=static_cast<float>(i+1);
    r.bind_data("X", xv); r.bind_data("dY", dyd); r.bind_data("dC", dcd);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("dC");
    nntile::tile::Tile<fp32_t> TX(xh), DY(dyh), DC(dch);
    using Yf = typename fp32_t::repr_t;
    { auto a=TX.acquire(STARPU_W),b=DY.acquire(STARPU_W),c=DC.acquire(STARPU_W);
      for(Index i=0;i<nx;++i) a[i]=Yf(static_cast<float>(i+1));
      for(Index i=0;i<4;++i) { b[i]=Yf(static_cast<float>(i+1)); c[i]=Yf(0.0f); }
      a.release();b.release();c.release(); }
    nntile::tile::conv2d_bwd_weight_inplace<fp32_t>(3,3,1,1,2,2,1,1,1,0,0,1.0,TX,DY,2,2,1,1,0.0,DC);
    starpu_task_wait_for_all();
    std::vector<float> tr(4);
    { auto L=DC.acquire(STARPU_R);
      for(Index i=0;i<4;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(Index i=0;i<4;++i) REQUIRE(std::abs(gout[static_cast<size_t>(i)]-tr[static_cast<size_t>(i)])<1e+2f);
}