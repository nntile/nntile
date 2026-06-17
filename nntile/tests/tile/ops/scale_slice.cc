/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/scale_slice.cc
 * Test TileGraph scale slice vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/scale_slice.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/scale_slice.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph scale_slice", "[graph][tile]")
{
    const std::vector<Index> stor_t1s = {3, 5};
    const std::vector<Index> graph_t1s = graph_shape(stor_t1s);
    const std::vector<Index> stor_t2s = {3, 4, 5};
    const std::vector<Index> graph_t2s = graph_shape(stor_t2s);
    const Index n1 = 15, n2 = 60;
    const Scalar a = 0.75;
    const Index stor_axis = 1;
    const Index g_axis =
        graph_axis(stor_axis, static_cast<Index>(stor_t2s.size()));
    TileGraph g("g");
    auto* t1 = g.data(graph_t1s, "t1", DataType::FP32);
    auto* t2 = g.data(graph_t2s, "t2", DataType::FP32);
    t1->mark_input(true);
    t2->mark_output(true);
    tg::scale_slice(a, t1, t2, g_axis);
    Runtime rt(g);
    rt.compile();
    std::vector<float> v1(n1), v2(n2, 0.f);
    for(Index i = 0; i < n1; ++i) { v1[static_cast<size_t>(i)] = static_cast<float>(i) + 0.5f; }
    rt.bind_data(t1, v1);
    rt.bind_data(t2, v2);
    rt.execute();
    rt.wait();
    const std::vector<float> gout = rt.get_output<float>(t2);
    nntile::core::Tile<fp32_t> T1(stor_t1s), T2(stor_t2s);
    using Y = typename nntile::fp32_t::repr_t;
    { auto A = T1.acquire(STARPU_W), B = T2.acquire(STARPU_W);
      for(Index i = 0; i < n1; ++i) A[i] = Y(v1[static_cast<size_t>(i)]);
      for(Index i = 0; i < n2; ++i) B[i] = Y(0);
      A.release(); B.release(); }
    nntile::core::scale_slice<fp32_t>(-1, a, T1, T2, stor_axis);
    starpu_task_wait_for_all();
    std::vector<float> tref(60);
    { auto L = T2.acquire(STARPU_R);
      for(Index i = 0; i < 60; ++i) tref[static_cast<size_t>(i)] = static_cast<float>(L[i]);
      L.release(); }
    for(size_t i = 0; i < tref.size(); ++i) REQUIRE(std::abs(gout[i] - tref[i]) < 1e-2f);
}
