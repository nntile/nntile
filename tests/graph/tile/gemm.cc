/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/gemm.cc
 * Test TileGraph gemm vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/gemm.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/gemm.hh"
#include "nntile/tile/tile.hh"
#include "nntile/constants.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph gemm matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 2};
    const Index nelems = 4;
    TileGraph g("g");
    auto* a = g.data(sh, "a", DataType::FP32);
    auto* b = g.data(sh, "b", DataType::FP32);
    auto* c = g.data(sh, "c", DataType::FP32);
    a->mark_input(true);
    b->mark_input(true);
    c->mark_input(true);
    c->mark_output(true);
    const Scalar alpha = 1.0, beta = 0.0;
    const bool trans_a = false, trans_b = false;
    const Index ndim = 1;
    const Index batch_ndim = 0;
    tg::gemm(a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> av(nelems), bv(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        av[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        bv[static_cast<size_t>(i)] = 0.2f * static_cast<float>(i + 1);
    }
    std::vector<float> cv(nelems, 0.f);
    runtime.bind_data("a", av);
    runtime.bind_data("b", bv);
    runtime.bind_data("c", cv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("c");
    nntile::tile::Tile<fp32_t> ta(sh), tb(sh), tc(sh);
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
    nntile::tile::gemm<fp32_t>(alpha, opN, ta, opN, tb, beta, tc, ndim, batch_ndim, 0);
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
