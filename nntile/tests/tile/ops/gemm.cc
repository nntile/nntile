/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/gemm.cc
 * Test TileGraph gemm vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/gemm.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/gemm.hh"
#include "nntile/core/tile.hh"
#include "nntile/constants.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph gemm matches tile", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {2, 2};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const Index nelems = 4;
    TileGraph g("g");
    auto* a = g.data(graph_sh, "a", DataType::FP32);
    auto* b = g.data(graph_sh, "b", DataType::FP32);
    auto* c = g.data(graph_sh, "c", DataType::FP32);
    a->mark_input(true);
    b->mark_input(true);
    c->mark_input(true);
    c->mark_output(true);
    const Scalar alpha = 1.0, beta = 0.0;
    const bool trans_a = false, trans_b = false;
    const Index ndim = 1;
    const Index batch_ndim = 0;
    tg::gemm(a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> av(nelems), bv(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        av[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        bv[static_cast<size_t>(i)] = 0.2f * static_cast<float>(i + 1);
    }
    std::vector<float> cv(nelems, 0.f);
    runtime.bind_data(a, av);
    runtime.bind_data(b, bv);
    runtime.bind_data(c, cv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(c);
    nntile::core::Tile<fp32_t> ta(stor_sh), tb(stor_sh), tc(stor_sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = ta.acquire(STARPU_W);
        auto l2 = tb.acquire(STARPU_W);
        auto l3 = tc.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            l1[i] = Y(av[static_cast<size_t>(i)]);
            l2[i] = Y(bv[static_cast<size_t>(i)]);
            l3[i] = Y(0);
        }
        l1.release();
        l2.release();
        l3.release();
    }
    const TransOp opN(TransOp::NoTrans);
    // Tile execute swaps operands/transposes for Fortran storage.
    nntile::core::gemm<fp32_t>(
        -1, alpha, opN, tb, opN, ta, beta, tc, ndim, batch_ndim, 0);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = tc.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-3f;
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph gemm asymmetric shapes and batch_ndim",
    "[graph][tile]")
{
  // Graph GEMM (ndim=1): A {N,K}, B {K,M}, C {N,M}.
    const std::vector<Index> graph_a = {5, 4};
    const std::vector<Index> stor_a = storage_shape(graph_a);
    const std::vector<Index> graph_b = {4, 3};
    const std::vector<Index> stor_b = storage_shape(graph_b);
    const std::vector<Index> graph_c = {5, 3};
    const std::vector<Index> stor_c = storage_shape(graph_c);
    const Index na = 20, nb = 12, nc = 15;
    const Scalar alpha = 1.0, beta = 0.0;
    const bool trans_a = false, trans_b = false;
    const Index ndim = 1, batch_ndim = 0;

    TileGraph g("g");
    auto* a = g.data(graph_a, "a", DataType::FP32);
    auto* b = g.data(graph_b, "b", DataType::FP32);
    auto* c = g.data(graph_c, "c", DataType::FP32);
    a->mark_input(true);
    b->mark_input(true);
    c->mark_input(true);
    c->mark_output(true);
    tg::gemm(a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);
    Runtime runtime(g);
    runtime.compile();

    std::vector<float> av(na), bv(nb), cv(nc, 0.f);
    for(Index i = 0; i < na; ++i)
    {
        av[static_cast<size_t>(i)] = 0.03f * static_cast<float>(i + 1);
    }
    for(Index i = 0; i < nb; ++i)
    {
        bv[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i + 1);
    }
    runtime.bind_data(a, av);
    runtime.bind_data(b, bv);
    runtime.bind_data(c, cv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(c);

    nntile::core::Tile<fp32_t> ta(stor_a), tb(stor_b), tc(stor_c);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto la = ta.acquire(STARPU_W);
        auto lb = tb.acquire(STARPU_W);
        auto lc = tc.acquire(STARPU_W);
        for(Index i = 0; i < na; ++i) { la[i] = Y(av[static_cast<size_t>(i)]); }
        for(Index i = 0; i < nb; ++i) { lb[i] = Y(bv[static_cast<size_t>(i)]); }
        for(Index i = 0; i < nc; ++i) { lc[i] = Y(0); }
        la.release();
        lb.release();
        lc.release();
    }
    const TransOp opN(TransOp::NoTrans);
    nntile::core::gemm<fp32_t>(
        -1, alpha, opN, tb, opN, ta, beta, tc, ndim, batch_ndim, 0);
    starpu_task_wait_for_all();

    std::vector<float> tref(nc);
    {
        auto lc = tc.acquire(STARPU_R);
        for(Index i = 0; i < nc; ++i)
        {
            tref[static_cast<size_t>(i)] = static_cast<float>(lc[i]);
        }
        lc.release();
    }
    constexpr float tol = 1e-3f;
    for(size_t i = 0; i < tref.size(); ++i)
    {
        REQUIRE(std::abs(gout[i] - tref[i]) < tol);
    }

    // Batched graph GEMM (batch_ndim=1): {B,N,K} @ {B,K,M} -> {B,N,M}.
    const std::vector<Index> graph_ba = {2, 5, 4};
    const std::vector<Index> stor_ba = storage_shape(graph_ba);
    const std::vector<Index> graph_bb = {2, 4, 3};
    const std::vector<Index> stor_bb = storage_shape(graph_bb);
    const std::vector<Index> graph_bc = {2, 5, 3};
    const std::vector<Index> stor_bc = storage_shape(graph_bc);
    const Index nba = 40, nbb = 24, nbc = 30;
    const Index batch_nd = 1;

    TileGraph g2("g2");
    auto* ba = g2.data(graph_ba, "a", DataType::FP32);
    auto* bb = g2.data(graph_bb, "b", DataType::FP32);
    auto* bc = g2.data(graph_bc, "c", DataType::FP32);
    ba->mark_input(true);
    bb->mark_input(true);
    bc->mark_input(true);
    bc->mark_output(true);
    tg::gemm(ba, bb, bc, alpha, beta, trans_a, trans_b, ndim, batch_nd);
    Runtime runtime2(g2);
    runtime2.compile();

    std::vector<float> bav(nba), bbv(nbb), bcv(nbc, 0.f);
    for(Index i = 0; i < nba; ++i)
    {
        bav[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
    }
    for(Index i = 0; i < nbb; ++i)
    {
        bbv[static_cast<size_t>(i)] = 0.015f * static_cast<float>(i + 1);
    }
    runtime2.bind_data(ba, bav);
    runtime2.bind_data(bb, bbv);
    runtime2.bind_data(bc, bcv);
    runtime2.execute();
    runtime2.wait();
    const std::vector<float> gout2 = runtime2.get_output<float>(bc);

    nntile::core::Tile<fp32_t> tba(stor_ba), tbb(stor_bb), tbc(stor_bc);
    {
        auto la = tba.acquire(STARPU_W);
        auto lb = tbb.acquire(STARPU_W);
        auto lc = tbc.acquire(STARPU_W);
        for(Index i = 0; i < nba; ++i) { la[i] = Y(bav[static_cast<size_t>(i)]); }
        for(Index i = 0; i < nbb; ++i) { lb[i] = Y(bbv[static_cast<size_t>(i)]); }
        for(Index i = 0; i < nbc; ++i) { lc[i] = Y(0); }
        la.release();
        lb.release();
        lc.release();
    }
    nntile::core::gemm<fp32_t>(
        -1, alpha, opN, tbb, opN, tba, beta, tbc, ndim, batch_nd, 0);
    starpu_task_wait_for_all();

    std::vector<float> tref2(nbc);
    {
        auto lc = tbc.acquire(STARPU_R);
        for(Index i = 0; i < nbc; ++i)
        {
            tref2[static_cast<size_t>(i)] = static_cast<float>(lc[i]);
        }
        lc.release();
    }
    for(size_t i = 0; i < tref2.size(); ++i)
    {
        REQUIRE(std::abs(gout2[i] - tref2[i]) < tol);
    }
}
