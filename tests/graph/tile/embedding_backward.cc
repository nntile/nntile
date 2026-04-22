/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/embedding_backward.cc
 * Test TileGraph embedding backward vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/embedding_backward.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/embedding_backward.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph embedding_backward", "[graph][tile]")
{
    const Index m=2,n=2,k=3,k0=0,ks=3; const int redux=0;
    const std::vector<Index> ih={m,n}, egh={m,k,n}, vh={ks,5};
    TileGraph g("g");
    auto* index = g.data(ih, "index", DataType::INT64);
    auto* eg = g.data(egh, "eg", DataType::FP32);
    auto* vg = g.data(vh, "vg", DataType::FP32);
    index->mark_input(true); eg->mark_input(true); vg->mark_input(true); vg->mark_output(true);
    tg::embedding_backward(m,n,k,k0,ks,index,eg,vg,redux);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<std::int64_t> iv(4);
    iv[0]=0; iv[1]=2; iv[2]=4; iv[3]=1;
    std::vector<float> ega(12), vga(15);
    for(int i=0;i<12;++i) ega[static_cast<size_t>(i)]=0.5f*static_cast<float>(i+1);
    for(int j=0;j<15;++j) vga[static_cast<size_t>(j)]=0.1f*static_cast<float>(j+1);
    r.bind_data("index", iv);
    r.bind_data("eg", ega);
    r.bind_data("vg", vga);
    r.execute(); r.wait();
    const auto gout = r.get_output<float>("vg");
    nntile::tile::Tile<nntile::int64_t> I(ih);
    nntile::tile::Tile<fp32_t> Eg(egh), Vg(vh);
    using Y = typename fp32_t::repr_t;
    { auto a=I.acquire(STARPU_W); a[0]=0; a[1]=2; a[2]=4; a[3]=1; a.release(); }
    { auto b=Eg.acquire(STARPU_W);
      for(int i=0;i<12;++i) b[i]=Y(ega[static_cast<size_t>(i)]);
      b.release(); }
    { auto c=Vg.acquire(STARPU_W);
      for(int j=0;j<15;++j) c[j]=Y(vga[static_cast<size_t>(j)]);
      c.release(); }
    nntile::tile::embedding_backward<fp32_t>(m,n,k,k0,ks,I,Eg,Vg,redux);
    starpu_task_wait_for_all();
    std::vector<float> tr(15);
    { auto L=Vg.acquire(STARPU_R);
      for(int j=0;j<15;++j) tr[static_cast<size_t>(j)]=static_cast<float>(L[j]);
      L.release(); }
    for(int j=0;j<15;++j) REQUIRE(std::abs(gout[static_cast<size_t>(j)]-tr[static_cast<size_t>(j)])<1e+2f);
}