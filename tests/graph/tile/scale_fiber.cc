/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/scale_fiber.cc
 * Test TileGraph scale fiber vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/scale_fiber.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/scale_fiber.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph scale_fiber", "[graph][tile]")
{
    const std::vector<Index> full = {3, 4, 5};
    const std::vector<Index> fib = {5};
    const Index n = 60, nf = 5;
    const Scalar a = 1.25;
    const Index axis = 2, batch = 0;
    TileGraph g("g");
    auto* s = g.data(fib, "s", DataType::FP32);
    auto* d = g.data(full, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::scale_fiber(a, s, d, axis, batch);
    TileGraph::Runtime rt(g);
    rt.compile();
    std::vector<float> f1(nf);
    std::vector<float> f2(n, 0.f);
    for(Index i = 0; i < nf; ++i) { f1[static_cast<size_t>(i)] = 0.5f + static_cast<float>(i); }
    rt.bind_data("s", f1);
    rt.bind_data("d", f2);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(fib), td(full);
    using Y = typename nntile::fp32_t::repr_t;
    { auto A = ts.acquire(STARPU_W), B = td.acquire(STARPU_W);
      for(Index i = 0; i < nf; ++i) A[i] = Y(f1[static_cast<size_t>(i)]);
      for(Index i = 0; i < n; ++i) B[i] = Y(0);
      A.release(); B.release(); }
    nntile::tile::scale_fiber<fp32_t>(a, ts, td, axis, batch);
    starpu_task_wait_for_all();
    std::vector<float> tref(n);
    { auto L = td.acquire(STARPU_R);
      for(Index i = 0; i < n; ++i) tref[static_cast<size_t>(i)] = static_cast<float>(L[i]);
      L.release(); }
    for(size_t i = 0; i < tref.size(); ++i) REQUIRE(std::abs(gout[i] - tref[i]) < 1e-3f);
}
