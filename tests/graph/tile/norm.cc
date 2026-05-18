/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/norm.cc
 * Test TileGraph norm vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/norm.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/norm.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph norm matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    const Scalar alpha = 1.0, beta = 0.0;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(std::vector<Index>{}, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::norm(alpha, s, beta, d);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i) - 0.1f; }
    runtime.bind_data("s", sv);
    std::vector<float> dv(1, 0.f);
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sh), td(std::vector<Index>{});
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = ts.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { a[i] = Y(sv[static_cast<size_t>(i)]); }
        a.release();
    }
    {
        auto b = td.acquire(STARPU_W);
        b[0] = Y(0);
        b.release();
    }
    nntile::tile::norm<fp32_t>(alpha, ts, beta, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(1);
    {
        auto l2 = td.acquire(STARPU_R);
        tref[0] = static_cast<float>(l2[0]);
        l2.release();
    }
    constexpr float tol = 1e-3f;
    REQUIRE(gout.size() == 1u);
    REQUIRE(std::abs(gout[0] - tref[0]) < tol);
}
