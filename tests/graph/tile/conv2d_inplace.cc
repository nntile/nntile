/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/conv2d_inplace.cc
 * Test TileGraph conv2d inplace vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/conv2d_inplace.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/conv2d_inplace.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph conv2d_inplace", "[graph][tile]")
{
    const std::vector<Index> xh={3,3,1,1}, ch={2,2,1,1}, yh={2,2,1,1};
    const Index nx=9, nc=4, ny=4;
    TileGraph g("g");
    auto* X = g.data(xh, "X", DataType::FP32);
    auto* C = g.data(ch, "C", DataType::FP32);
    auto* Y = g.data(yh, "Y", DataType::FP32);
    X->mark_input(true); C->mark_input(true); Y->mark_input(true); Y->mark_output(true);
    tg::conv2d_inplace(3,3,1,1,2,2,1,1,1,0,0,1.0,X,C,2,2,1,1,0.0,Y);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> xv(nx), cv(nc), yv(ny,0.f);
    for(Index i=0;i<nx;++i) xv[static_cast<size_t>(i)]=static_cast<float>(i+1);
    for(Index i=0;i<nc;++i) cv[static_cast<size_t>(i)]=static_cast<float>(i+1);
    r.bind_data("X", xv); r.bind_data("C", cv); r.bind_data("Y", yv);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("Y");
    nntile::tile::Tile<fp32_t> TX(xh), TC(ch), TY(yh);
    using Yf = typename fp32_t::repr_t;
    { auto a=TX.acquire(STARPU_W),b=TC.acquire(STARPU_W),c=TY.acquire(STARPU_W);
      for(Index i=0;i<nx;++i) a[i]=Yf(xv[static_cast<size_t>(i)]);
      for(Index i=0;i<nc;++i) b[i]=Yf(cv[static_cast<size_t>(i)]);
      for(Index i=0;i<ny;++i) c[i]=Yf(0.0f);
      a.release();b.release();c.release(); }
    nntile::tile::conv2d_inplace<fp32_t>(3,3,1,1,2,2,1,1,1,0,0,1.0,TX,TC,2,2,1,1,0.0,TY);
    starpu_task_wait_for_all();
    std::vector<float> tr(4);
    { auto L=TY.acquire(STARPU_R);
      for(Index i=0;i<4;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(Index i=0;i<4;++i) REQUIRE(std::abs(gout[static_cast<size_t>(i)]-tr[static_cast<size_t>(i)])<1e+2f);
}
