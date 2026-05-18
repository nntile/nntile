/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/add_inplace.cc
 * Test TileGraph add inplace vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/add_inplace.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/add_inplace.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph add_inplace matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    const Scalar alpha = 2.0, beta = 1.0;
    TileGraph g("g");
    auto* x = g.data(sh, "x", DataType::FP32);
    auto* y = g.data(sh, "y", DataType::FP32);
    x->mark_input(true);
    y->mark_input(true);
    y->mark_output(true);
    tg::add_inplace(alpha, x, beta, y);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> xv(nelems), yv(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        xv[static_cast<size_t>(i)] = static_cast<float>(i);
        yv[static_cast<size_t>(i)] = static_cast<float>(10 * i);
    }
    runtime.bind_data("x", xv);
    runtime.bind_data("y", yv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("y");
    nntile::tile::Tile<fp32_t> tx(sh), ty(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto a = tx.acquire(STARPU_W);
        auto b = ty.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            a[i] = Y(static_cast<float>(i));
            b[i] = Y(static_cast<float>(10 * i));
        }
        a.release();
        b.release();
    }
    nntile::tile::add_inplace<fp32_t>(alpha, tx, beta, ty);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = ty.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-3f;
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
