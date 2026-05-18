/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/hypot.cc
 * Test TileGraph hypot vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/hypot.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/hypot.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph hypot matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    const Scalar alpha = 1.0, beta = 1.0;
    TileGraph g("g");
    auto* s1 = g.data(sh, "s1", DataType::FP32);
    auto* s2 = g.data(sh, "s2", DataType::FP32);
    auto* d = g.data(sh, "d", DataType::FP32);
    s1->mark_input(true);
    s2->mark_input(true);
    d->mark_output(true);
    tg::hypot(alpha, s1, beta, s2, d);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> v1(nelems), v2(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        v1[static_cast<size_t>(i)] = 0.2f * static_cast<float>(i) + 0.1f;
        v2[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i) + 0.2f;
    }
    runtime.bind_data("s1", v1);
    runtime.bind_data("s2", v2);
    std::vector<float> dv(nelems, 0.f);
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> t1(sh), t2(sh), td(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = t1.acquire(STARPU_W);
        auto b = t2.acquire(STARPU_W);
        auto c = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            a[i] = Y(v1[static_cast<size_t>(i)]);
            b[i] = Y(v2[static_cast<size_t>(i)]);
            c[i] = Y(0);
        }
        a.release();
        b.release();
        c.release();
    }
    nntile::tile::hypot<fp32_t>(alpha, t1, beta, t2, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-4f;
    REQUIRE(gout.size() == tref.size());
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
