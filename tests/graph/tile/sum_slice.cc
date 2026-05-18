/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/sum_slice.cc
 * Test TileGraph sum slice vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/sum_slice.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/sum_slice.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph sum_slice (axis=0)", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 4, 5};
    const std::vector<Index> dh = {4, 5};
    const Index nelems = 60, nd = 4 * 5;
    const Scalar a = 1.0, b = 0.0;
    const Index axis = 0;
    const int redux = 0;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(dh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::sum_slice(a, s, b, d, axis, redux);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    std::vector<float> dd(nd, 0.f);
    runtime.bind_data("s", sv);
    runtime.bind_data("d", dd);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> ts(sh), td(dh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto A = ts.acquire(STARPU_W);
        auto B = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { A[i] = Y(sv[static_cast<size_t>(i)]); }
        for(Index j = 0; j < nd; ++j) { B[j] = Y(0); }
        A.release();
        B.release();
    }
    nntile::tile::sum_slice<fp32_t>(a, ts, b, td, axis, redux);
    starpu_task_wait_for_all();
    std::vector<float> tref(nd);
    {
        auto L = td.acquire(STARPU_R);
        for(Index j = 0; j < nd; ++j) { tref[static_cast<size_t>(j)] = static_cast<float>(L[j]); }
        L.release();
    }
    for(size_t j = 0; j < tref.size(); ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < 1e-1f); }
}
