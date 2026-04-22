/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/copy_intersection.cc
 * Test TileGraph copy intersection vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/copy_intersection.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/copy_intersection.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph copy_intersection", "[graph][tile]")
{
    const std::vector<Index> sh = {2,2,3};
    const std::vector<Index> sc = {6};
    const Index n = 12;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(sh, "d", DataType::FP32);
    auto* scra = g.data(sc, "scratch", DataType::INT64);
    s->mark_input(true);
    d->mark_input(true);
    d->mark_output(true);
    scra->mark_input(true);
    tg::copy_intersection(s, {0,0,0}, d, {0,0,0}, scra);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> sv(n), dv(n, 0.f);
    for(Index i=0;i<n;++i) sv[static_cast<size_t>(i)]=static_cast<float>(i+1);
    std::vector<std::int64_t> scv(6, 0);
    r.bind_data("s", sv);
    r.bind_data("d", dv);
    r.bind_data("scratch", scv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<fp32_t> S(sh), D(sh);
    nntile::tile::Tile<nntile::int64_t> Sc(sc);
    using Y = typename fp32_t::repr_t;
    { auto a=S.acquire(STARPU_W), b=D.acquire(STARPU_W);
      for(Index i=0;i<n;++i) { a[i]=Y(sv[static_cast<size_t>(i)]); b[i]=Y(0);} a.release(); b.release(); }
    { auto L=Sc.acquire(STARPU_W); for(Index j=0;j<6;++j) L[j]=0; L.release(); }
    nntile::tile::copy_intersection<fp32_t>(
        S, {0, 0, 0}, D, {0, 0, 0}, Sc);
    starpu_task_wait_for_all();
    std::vector<float> tr(n);
    { auto L=D.acquire(STARPU_R);
      for(Index i=0;i<n;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(size_t i=0;i<tr.size();++i) REQUIRE(std::abs(gout[i]-tr[i])<1e-4f);
}