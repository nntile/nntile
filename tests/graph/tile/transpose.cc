/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/transpose.cc
 * Test TileGraph transpose vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/transpose.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/transpose.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph transpose matches tile", "[graph][tile]")
{
    const std::vector<Index> sshape = {3, 5};
    const std::vector<Index> dshape = {5, 3};
    const Index nelems = 3 * 5;
    const Scalar alpha = 0.5;
    const Index ndim = 1;
    TileGraph g("g");
    auto* s = g.data(sshape, "s", DataType::FP32);
    auto* d = g.data(dshape, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::transpose(alpha, s, d, ndim);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i); }
    runtime.bind_data("s", sv);
    std::vector<float> dv(15, 1.f);
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sshape), td(dshape);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = ts.acquire(STARPU_W);
        auto b = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { a[i] = Y(sv[static_cast<size_t>(i)]); }
        for(Index i = 0; i < 15; ++i) { b[i] = Y(1.f); }
        a.release();
        b.release();
    }
    nntile::tile::transpose<fp32_t>(alpha, ts, td, ndim);
    starpu_task_wait_for_all();
    std::vector<float> tref(15);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < 15; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-4f;
    REQUIRE(gout.size() == tref.size());
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
