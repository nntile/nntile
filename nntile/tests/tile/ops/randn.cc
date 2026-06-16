/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/randn.cc
 * Test TileGraph randn vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/randn.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/randn.hh"
#include "nntile/core/tile.hh"
using namespace nntile; using namespace nntile; namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph randn", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {3,4,5};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const std::vector<Index> st = {1,1,1}, us = {5,6,7};
    const unsigned long long seed = static_cast<unsigned long long>(-1);
    const Scalar mean = 1.0, std = 2.0;
    TileGraph g("g");
    auto* d = g.data(graph_sh, "d", DataType::FP32);
    d->mark_input(true);
    d->mark_output(true);
    tg::randn(d, st, us, seed, mean, std);
    Runtime r(g);
    r.compile();
    std::vector<float> dv(60, 0.f);
    r.bind_data(d, dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<fp32_t> Td(stor_sh);
    nntile::core::randn<fp32_t>(-1, Td, st, us, seed, mean, std);
    starpu_task_wait_for_all();
    std::vector<float> tref(60);
    { auto L=Td.acquire(STARPU_R);
      for(Index i=0;i<60;++i) tref[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    for(size_t i=0;i<60;++i) REQUIRE(gout[i]==tref[i]);
}
