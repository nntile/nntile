/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/norm_slice.cc
 * Test TileGraph norm slice vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/norm_slice.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/norm_slice.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph norm_slice (axis=0)", "[graph][tile]")
{
    const std::vector<Index> t1h = {3, 4, 5}, t2h = {4, 5}, dh = {4, 5};
    const Index n1 = 60, n2 = 20, n3 = 20;
    const Scalar a = -1.0, b = 0.5;
    const Index ax = 0;
    const int redux = 0;
    TileGraph g("g");
    auto* t1 = g.data(t1h, "t1", DataType::FP32);
    auto* t2 = g.data(t2h, "t2", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    t1->mark_input(true);
    t2->mark_input(true);
    d->mark_output(true);
    tg::norm_slice(a, t1, b, t2, d, ax, redux);
    TileGraph::Runtime rt(g);
    rt.compile();
    std::vector<float> u1(n1), u2(n2, 0.f), o(n3, 0.f);
    for(Index i = 0; i < n1; ++i) { u1[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    rt.bind_data("t1", u1);
    rt.bind_data("t2", u2);
    rt.bind_data("d", o);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>("d");
    nntile::tile::Tile<fp32_t> T1(t1h), T2(t2h), D(dh);
    using Y = typename nntile::fp32_t::repr_t;
    { auto A = T1.acquire(STARPU_W), B = T2.acquire(STARPU_W), C = D.acquire(STARPU_W);
      for(Index i = 0; i < n1; ++i) A[i] = Y(u1[static_cast<size_t>(i)]);
      for(Index j = 0; j < n2; ++j) { B[j] = Y(0); C[j] = Y(0); }
      A.release(); B.release(); C.release(); }
    nntile::tile::norm_slice<fp32_t>(a, T1, b, T2, D, ax, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(20);
    { auto L = D.acquire(STARPU_R);
      for(Index j = 0; j < 20; ++j) tref[static_cast<size_t>(j)] = static_cast<float>(L[j]);
      L.release(); }
    for(size_t j = 0; j < tref.size(); ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < 1e+2f); }
}
