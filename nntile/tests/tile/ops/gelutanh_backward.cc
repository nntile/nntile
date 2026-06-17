/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/gelutanh_backward.cc
 * Test TileGraph gelutanh backward vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/gelutanh_backward.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/gelutanh_backward.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph gelutanh_backward matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 2};
    const Index nelems = 6;
    TileGraph g("g");
    auto* x = g.data(sh, "x", DataType::FP32);
    auto* dy = g.data(sh, "dy", DataType::FP32);
    auto* dx = g.data(sh, "dx", DataType::FP32);
    x->mark_input(true);
    dy->mark_input(true);
    dx->mark_input(true);
    dx->mark_output(true);
    tg::gelutanh_backward(x, dy, dx);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> xv(nelems), dyv(nelems), dxv(nelems, 0.f);
    for(Index i = 0; i < nelems; ++i)
    {
        xv[static_cast<size_t>(i)] = 0.3f * static_cast<float>(i) - 0.4f;
        dyv[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i) + 0.05f;
    }
    runtime.bind_data(x, xv);
    runtime.bind_data(dy, dyv);
    runtime.bind_data(dx, dxv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(dx);
    nntile::core::Tile<fp32_t> tx(sh), tdy(sh), tdx(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = tx.acquire(STARPU_W);
        auto l2 = tdy.acquire(STARPU_W);
        auto l3 = tdx.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            l1[i] = Y(xv[static_cast<size_t>(i)]);
            l2[i] = Y(dyv[static_cast<size_t>(i)]);
            l3[i] = Y(0);
        }
        l1.release();
        l2.release();
        l3.release();
    }
    nntile::core::gelutanh_backward<fp32_t>(-1, tx, tdy, tdx);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = tdx.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    nntile::test::require_relative_element_error(gout, tref);
}
