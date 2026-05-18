/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/mask_scalar.cc
 * Test TileGraph mask scalar vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/mask_scalar.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/mask_scalar.hh"
#include "nntile/tile/tile.hh"
#include <array>
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph mask_scalar", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index n = 6;
    const Scalar val = -9.0;
    const Index batch = 0;
    TileGraph g("g");
    auto* mask = g.data(sh, "mask", DataType::BOOL);
    auto* a = g.data(sh, "a", DataType::FP32);
    mask->mark_input(true);
    a->mark_input(true);
    a->mark_output(true);
    tg::mask_scalar(mask, val, a, batch);
    TileGraph::Runtime r(g);
    r.compile();
    std::array<bool, 6> mb{};
    for(Index i = 0; i < n; ++i) { mb[static_cast<size_t>(i)] = (static_cast<int>(i) % 3) != 0; }
    std::vector<float> av(n);
    for(Index i = 0; i < n; ++i) { av[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    r.bind_data("mask", mb.data(), static_cast<size_t>(n));
    r.bind_data("a", av);
    r.execute();
    r.wait();
    const std::vector<float> gout = r.get_output<float>("a");
    nntile::tile::Tile<bool_t> Tm(sh);
    nntile::tile::Tile<fp32_t> Ta(sh);
    using Y = typename fp32_t::repr_t;
    { auto mloc = Tm.acquire(STARPU_W);
      for(Index i = 0; i < n; ++i) { mloc[i] = nntile::bool_t(mb[static_cast<size_t>(i)]); }
      mloc.release(); }
    { auto aloc = Ta.acquire(STARPU_W);
      for(Index i = 0; i < n; ++i) { aloc[i] = Y(static_cast<float>(i + 1)); }
      aloc.release(); }
    nntile::tile::mask_scalar<fp32_t>(Tm, val, Ta, batch);
    starpu_task_wait_for_all();
    std::vector<float> tref(n);
    { auto L = Ta.acquire(STARPU_R);
      for(Index i = 0; i < n; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(L[i]); }
      L.release(); }
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < 1e-4f); }
}
