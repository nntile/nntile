/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/clear.cc
 * Test TileGraph clear vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/clear.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph clear matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    TileGraph g("g");
    auto* x = g.data(sh, "x", DataType::FP32);
    x->mark_input(true);
    x->mark_output(true);
    tg::clear(x);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> xv(nelems);
    for(Index i = 0; i < nelems; ++i) { xv[static_cast<size_t>(i)] = static_cast<float>(i) + 1.5f; }
    runtime.bind_data("x", xv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("x");
    nntile::tile::Tile<fp32_t> tx(sh);
    {
        using Y = typename nntile::fp32_t::repr_t;
        auto l1 = tx.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(static_cast<float>(i) + 1.5f); }
        l1.release();
    }
    nntile::tile::clear<fp32_t>(tx);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = tx.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-4f;
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
