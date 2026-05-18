/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/silu.cc
 * TileGraph silu vs nntile::tile::silu (small parity B).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include "context_fixture.hh"
#include "nntile/graph/tile/silu.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/silu.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph silu matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(sh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::silu(s, d);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 0.2f; }
    runtime.bind_data("s", sv);
    std::vector<float> dv(nelems, 0.f);
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sh), td(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = ts.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(sv[static_cast<size_t>(i)]); }
        l1.release();
    }
    nntile::tile::silu<fp32_t>(ts, td);
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
