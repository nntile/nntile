/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/norm_fiber.cc
 * Test TileGraph norm fiber vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/norm_fiber.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/norm_fiber.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph norm_fiber", "[graph][tile]")
{
    const std::vector<Index> s1h = {5, 3, 20, 1};
    const std::vector<Index> s2h = {5};
    const std::vector<Index> dh = {5};
    const Scalar a = 1.0, b = 0.0;
    const Index ax = 0, bd = 0;
    const int redux = 0;
    const Index n1 = 5 * 3 * 20, n2 = 5, n3 = 5;
    TileGraph g("g");
    auto* t1 = g.data(s1h, "s1", DataType::FP32);
    auto* t2 = g.data(s2h, "s2", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    t1->mark_input(true);
    t2->mark_input(true);
    d->mark_output(true);
    tg::norm_fiber(a, t1, b, t2, d, ax, bd, redux);
    TileGraph::Runtime rt(g);
    rt.compile();
    std::vector<float> v1(n1), v2(n2);
    std::vector<float> v3(n3, 0.f);
    for(Index i = 0; i < n1; ++i) { v1[static_cast<size_t>(i)] = -1.0f; }
    for(Index i = 0; i < n2; ++i) { v2[static_cast<size_t>(i)] = 0.f; }
    rt.bind_data("s1", v1);
    rt.bind_data("s2", v2);
    rt.bind_data("d", v3);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>("d");
    nntile::tile::Tile<fp32_t> T1(s1h), T2(s2h), D(dh);
    using Y = typename nntile::fp32_t::repr_t;
    { auto a1 = T1.acquire(STARPU_W), a2 = T2.acquire(STARPU_W), a3 = D.acquire(STARPU_W);
      for(Index i = 0; i < n1; ++i) a1[i] = Y(-1.0f);
      for(Index i = 0; i < n2; ++i) a2[i] = Y(0.0f);
      for(Index i = 0; i < n3; ++i) a3[i] = Y(0.0f);
      a1.release(); a2.release(); a3.release(); }
    nntile::tile::norm_fiber<fp32_t>(a, T1, b, T2, D, ax, bd, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(5);
    { auto L = D.acquire(STARPU_R);
      for(Index j = 0; j < 5; ++j) tref[static_cast<size_t>(j)] = static_cast<float>(L[j]);
      L.release(); }
    for(int j = 0; j < 5; ++j) { REQUIRE(std::abs(gout[static_cast<size_t>(j)] - tref[static_cast<size_t>(j)]) < 1e-1f); }
}
