/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/sumprod_slice.cc
 * Test TileGraph sumprod slice vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/sumprod_slice.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/sumprod_slice.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph sumprod_slice (axis=0)", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 4, 5};
    const std::vector<Index> dh = {4, 5};
    const Index n = 60, nd = 4 * 5;
    const Scalar a = 1.0, b = 0.0;
    const Index ax = 0;
    const int redux = 0;
    TileGraph g("g");
    auto* s1 = g.data(sh, "s1", DataType::FP32);
    auto* s2 = g.data(sh, "s2", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    s1->mark_input(true);
    s2->mark_input(true);
    d->mark_output(true);
    tg::sumprod_slice(a, s1, s2, b, d, ax, redux);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> v1(n), v2(n), dd(nd, 0.f);
    for(Index i = 0; i < n; ++i)
    {
        v1[static_cast<size_t>(i)] = static_cast<float>(i + 1) / static_cast<float>(i * i + 1);
        v2[static_cast<size_t>(i)] = static_cast<float>(i * i + 1);
    }
    runtime.bind_data("s1", v1);
    runtime.bind_data("s2", v2);
    runtime.bind_data("d", dd);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> t1(sh), t2(sh), td(dh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto A = t1.acquire(STARPU_W);
        auto B = t2.acquire(STARPU_W);
        auto C = td.acquire(STARPU_W);
        for(Index i = 0; i < n; ++i)
        {
            A[i] = Y(v1[static_cast<size_t>(i)]);
            B[i] = Y(v2[static_cast<size_t>(i)]);
        }
        for(Index j = 0; j < nd; ++j) { C[j] = Y(0); }
        A.release();
        B.release();
        C.release();
    }
    nntile::tile::sumprod_slice<fp32_t>(a, t1, t2, b, td, ax, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(nd);
    {
        auto L = td.acquire(STARPU_R);
        for(Index j = 0; j < nd; ++j) { tref[static_cast<size_t>(j)] = static_cast<float>(L[j]); }
        L.release();
    }
    for(size_t j = 0; j < tref.size(); ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < 1e+2f); }
}
