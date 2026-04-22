/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/softmax_inplace.cc
 * Test TileGraph softmax inplace vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/softmax_inplace.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/softmax_inplace.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph softmax_inplace axis0", "[graph][tile]")
{
    const std::vector<Index> mh = {2,4,5}, dh = {3,4,5};
    const Index nms=40, n=60; const Scalar al=1.0; const Index axis=0;
    TileGraph g("g");
    auto* m = g.data(mh, "m", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    m->mark_input(true); d->mark_input(true); d->mark_output(true);
    tg::softmax_inplace(m, al, d, axis);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> mv(nms), dd(n);
    for(Index j=0;j<nms;j+=2){ mv[static_cast<size_t>(j)]=static_cast<float>(j+1);
      mv[static_cast<size_t>(j+1)]=std::exp(static_cast<float>(j+2)/10.f);} 
    for(Index i=0;i<n;++i) dd[static_cast<size_t>(i)]=static_cast<float>(i+1);
    r.bind_data("m", mv); r.bind_data("d", dd);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<fp32_t> M(mh), D(dh);
    using Y = typename fp32_t::repr_t;
    { auto a=M.acquire(STARPU_W),c=D.acquire(STARPU_W);
      for(Index j=0;j<nms;j+=2) {a[j]=Y(mv[static_cast<size_t>(j)]); a[j+1]=Y(mv[static_cast<size_t>(j+1)]);}
      for(Index i=0;i<n;++i) c[i]=Y(static_cast<float>(i+1));
      a.release(); c.release(); }
    nntile::tile::softmax_inplace<fp32_t>(M, al, D, axis);
    starpu_task_wait_for_all();
    std::vector<float> tr(n);
    { auto L=D.acquire(STARPU_R); for(Index i=0;i<n;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]); L.release(); }
    for(size_t i=0;i<tr.size();++i) REQUIRE(std::abs(gout[i]-tr[i])<1e+2f);
}