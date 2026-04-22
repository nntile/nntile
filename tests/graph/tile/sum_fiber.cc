/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/sum_fiber.cc
 * Test TileGraph sum fiber vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/sum_fiber.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/sum_fiber.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph sum_fiber matches tile (axis=0)", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 4, 5};
    const std::vector<Index> dh = {3};
    const Index nelems = 3 * 4 * 5;
    const Scalar alpha = 1.0, beta = 0.0;
    const Index axis = 0, batch_ndim = 0, redux = 0;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::sum_fiber(alpha, s, beta, d, axis, batch_ndim, redux);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    std::vector<float> dd(3, 0.f);
    runtime.bind_data("s", sv);
    runtime.bind_data("d", dd);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sh), td(dh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = ts.acquire(STARPU_W);
        auto b = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { a[i] = Y(sv[static_cast<size_t>(i)]); }
        for(Index j = 0; j < 3; ++j) { b[j] = Y(0); }
        a.release();
        b.release();
    }
    nntile::tile::sum_fiber<fp32_t>(alpha, ts, beta, td, axis, batch_ndim, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(3);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index j = 0; j < 3; ++j) { tref[static_cast<size_t>(j)] = static_cast<float>(l2[j]); }
        l2.release();
    }
    constexpr float tol = 1e-2f;
    for(size_t j = 0; j < 3; ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < tol); }
}
