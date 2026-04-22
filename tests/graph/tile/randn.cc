/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/randn.cc
 * Test TileGraph randn vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/randn.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/tile.hh"
using namespace nntile; using namespace nntile::graph; namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph randn", "[graph][tile]")
{
    const std::vector<Index> sh = {3,4,5};
    const std::vector<Index> st = {1,1,1}, us = {5,6,7};
    const unsigned long long seed = static_cast<unsigned long long>(-1);
    const Scalar mean = 1.0, std = 2.0;
    TileGraph g("g");
    auto* d = g.data(sh, "d", DataType::FP32);
    d->mark_input(true);
    d->mark_output(true);
    tg::randn(d, st, us, seed, mean, std);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> dv(60, 0.f);
    r.bind_data("d", dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>("d");
    nntile::tile::Tile<fp32_t> Td(sh);
    nntile::tile::randn<fp32_t>(Td, st, us, seed, mean, std);
    starpu_task_wait_for_all();
    std::vector<float> tref(60);
    { auto L=Td.acquire(STARPU_R);
      for(Index i=0;i<60;++i) tref[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(size_t i=0;i<60;++i) REQUIRE(gout[i]==tref[i]);
}