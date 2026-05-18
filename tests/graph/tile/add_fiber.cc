/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/add_fiber.cc
 * Test TileGraph add fiber vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/add_fiber.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/add_fiber.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph add_fiber matches tile", "[graph][tile]")
{
    const std::vector<Index> full = {3, 4, 5};
    const std::vector<Index> fib = {5};
    const Index nfull = 3 * 4 * 5;
    const Index nfib = 5;
    const Index axis = 2;
    const Index batch = 0;
    const Scalar alpha = 1.0, beta = 1.0;
    TileGraph g("g");
    auto* s1 = g.data(fib, "s1", DataType::FP32);
    auto* s2 = g.data(full, "s2", DataType::FP32);
    auto* d = g.data(full, "d", DataType::FP32);
    s1->mark_input(true);
    s2->mark_input(true);
    d->mark_output(true);
    tg::add_fiber(alpha, s1, beta, s2, d, axis, batch);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> f1(nfib), f2(nfull), f3(nfull, 0.f);
    for(Index i = 0; i < nfib; ++i) { f1[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    for(Index i = 0; i < nfull; ++i) { f2[static_cast<size_t>(i)] = 0.5f * static_cast<float>(i + 1); }
    runtime.bind_data("s1", f1);
    runtime.bind_data("s2", f2);
    runtime.bind_data("d", f3);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> t1(fib), t2(full), dst(full);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = t1.acquire(STARPU_W);
        auto b = t2.acquire(STARPU_W);
        auto c = dst.acquire(STARPU_W);
        for(Index i = 0; i < nfib; ++i) { a[i] = Y(f1[static_cast<size_t>(i)]); }
        for(Index i = 0; i < nfull; ++i)
        {
            b[i] = Y(f2[static_cast<size_t>(i)]);
            c[i] = Y(0);
        }
        a.release();
        b.release();
        c.release();
    }
    nntile::tile::add_fiber<fp32_t>(alpha, t1, beta, t2, dst, axis, batch);
    starpu_task_wait_for_all();
    std::vector<float> tref(nfull);
    {
        auto l2 = dst.acquire(STARPU_R);
        for(Index i = 0; i < nfull; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-3f;
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
