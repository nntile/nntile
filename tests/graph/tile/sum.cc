/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/sum.cc
 * Test TileGraph sum vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/sum.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/sum.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph sum matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3, 4};
    const Index nelems = 2 * 3 * 4;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(std::vector<Index>{}, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    const Scalar alpha = 0.5;
    const Scalar beta = 0.0;
    tg::sum(alpha, s, beta, d);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = static_cast<float>(i) * 0.01f; }
    runtime.bind_data("s", sv);
    std::vector<float> dv(1, 0.f);
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sh);
    nntile::tile::Tile<fp32_t> td(std::vector<Index>{});
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = ts.acquire(STARPU_W);
        auto l2 = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(sv[static_cast<size_t>(i)]); }
        l2[0] = Y(0);
        l1.release();
        l2.release();
    }
    nntile::tile::sum<fp32_t>(alpha, ts, beta, td);
    starpu_task_wait_for_all();
    constexpr float tol = 1e-3f;
    REQUIRE(gout.size() == 1);
    {
        auto l2 = td.acquire(STARPU_R);
        REQUIRE(std::abs(gout[0] - static_cast<float>(l2[0])) < tol);
        l2.release();
    }
}
