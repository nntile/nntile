/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/subtract_indexed_outputs.cc
 * Test TileGraph subtract indexed outputs vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/subtract_indexed_outputs.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph subtract_indexed_outputs", "[graph][tile]")
{
    const std::vector<Index> lh = {2,2}, dh = {3,2,2};
    const Index nl = 4, nd = 3*2*2;
    const Scalar v = 0.5;
    const Index ign = -1;
    TileGraph g("g");
    auto* lab = g.data(lh, "labels", DataType::INT64);
    auto* d = g.data(dh, "d", DataType::FP32);
    lab->mark_input(true);
    d->mark_input(true);
    d->mark_output(true);
    tg::subtract_indexed_outputs(v, lab, d, ign);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<std::int64_t> lv(4);
    lv[0]=0; lv[1]=1; lv[2]=2; lv[3]=0;
    std::vector<float> dv(nd);
    for(Index i=0;i<nd;++i) dv[static_cast<size_t>(i)]=1.0f+0.1f*static_cast<float>(i);
    r.bind_data("labels", lv);
    r.bind_data("d", dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<nntile::int64_t> L(lh);
    nntile::tile::Tile<fp32_t> D(dh);
    using Y = typename fp32_t::repr_t;
    { auto a=L.acquire(STARPU_W);
      a[0]=0; a[1]=1; a[2]=2; a[3]=0; a.release(); }
    { auto b=D.acquire(STARPU_W);
      for(Index i=0;i<nd;++i) b[i]=Y(1.0f+0.1f*static_cast<float>(i));
      b.release(); }
    nntile::tile::subtract_indexed_outputs<fp32_t>(v, L, D, ign);
    starpu_task_wait_for_all();
    std::vector<float> tr(nd);
    { auto L2=D.acquire(STARPU_R);
      for(Index i=0;i<nd;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L2[i]);
      L2.release(); }
    for(size_t i=0;i<nd;++i) REQUIRE(std::abs(gout[i]-tr[i])<1e-4f);
}
