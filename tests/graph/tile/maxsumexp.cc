/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/maxsumexp.cc
 * Test TileGraph maxsumexp vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/maxsumexp.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/maxsumexp.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph maxsumexp axis0", "[graph][tile]")
{
    const std::vector<Index> sh = {3,4,5}, dh = {2,4,5};
    const Index n1 = 60, n2 = 2*4*5;
    const Index axis = 0; const int redux = 0;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    s->mark_input(true); d->mark_output(true);
    tg::maxsumexp(s, d, axis, redux);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> a(n1); std::vector<float> b(n2, 0.f);
    for(Index i=0;i<n1;++i) a[static_cast<size_t>(i)] = static_cast<float>(i+1);
    r.bind_data("s", a); r.bind_data("d", b);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<fp32_t> S(sh), D(dh);
    using Y = typename fp32_t::repr_t;
    { auto p=S.acquire(STARPU_W), q=D.acquire(STARPU_W);
      for(Index i=0;i<n1;++i) p[i]=Y(a[static_cast<size_t>(i)]);
      for(Index j=0;j<n2;++j) q[j]=Y(0);
      p.release(); q.release(); }
    nntile::tile::maxsumexp<fp32_t>(S, D, axis, redux);
    starpu_task_wait_for_all();
    std::vector<float> tr(n2);
    { auto L=D.acquire(STARPU_R);
      for(Index j=0;j<n2;++j) tr[static_cast<size_t>(j)]=static_cast<float>(L[j]);
      L.release(); }
    for(size_t j=0;j<tr.size();++j) REQUIRE(std::abs(gout[j]-tr[j])<1e+2f);
}
