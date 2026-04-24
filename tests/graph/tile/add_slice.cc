/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/add_slice.cc
 * Test TileGraph add slice vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/add_slice.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/add_slice.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph add_slice", "[graph][tile]")
{
    const std::vector<Index> t1s = {4, 5}, t2s = {3, 4, 5}, ds = {3, 4, 5};
    const Index n1 = 20, n2 = 60;
    const Scalar a = 0.5, b = 0.5;
    const Index axis = 0;
    TileGraph g("g");
    auto* t1 = g.data(t1s, "t1", DataType::FP32);
    auto* t2 = g.data(t2s, "t2", DataType::FP32);
    auto* d = g.data(ds, "d", DataType::FP32);
    t1->mark_input(true);
    t2->mark_input(true);
    d->mark_output(true);
    tg::add_slice(a, t1, b, t2, d, axis);
    TileGraph::Runtime rt(g);
    rt.compile();
    std::vector<float> v1(n1), v2(n2);
    for(Index i = 0; i < n1; ++i) { v1[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    for(Index i = 0; i < n2; ++i) { v2[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1); }
    std::vector<float> vo(60, 0.f);
    rt.bind_data("t1", v1);
    rt.bind_data("t2", v2);
    rt.bind_data("d", vo);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>("d");
    nntile::tile::Tile<fp32_t> T1(t1s), T2(t2s), D(ds);
    using Y = typename nntile::fp32_t::repr_t;
    { auto A = T1.acquire(STARPU_W), B = T2.acquire(STARPU_W), C = D.acquire(STARPU_W);
      for(Index i = 0; i < n1; ++i) A[i] = Y(v1[static_cast<size_t>(i)]);
      for(Index i = 0; i < n2; ++i) { B[i] = Y(v2[static_cast<size_t>(i)]); C[i] = Y(0); }
      A.release(); B.release(); C.release(); }
    nntile::tile::add_slice<fp32_t>(a, T1, b, T2, D, axis);
    starpu_task_wait_for_all();
    std::vector<float> tref(60);
    { auto L = D.acquire(STARPU_R);
      for(Index i = 0; i < 60; ++i) tref[static_cast<size_t>(i)] = static_cast<float>(L[i]);
      L.release(); }
    for(size_t i = 0; i < tref.size(); ++i) REQUIRE(std::abs(gout[i] - tref[i]) < 1e-2f);
}
