/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/copy.cc
 * TileGraph copy vs nntile::core::copy (small parity B).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/copy.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/copy.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph copy matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 2};
    const Index nelems = 6;
    TileGraph g("g");
    auto* s = g.data(sh, "s", DataType::FP32);
    auto* d = g.data(sh, "d", DataType::FP32);
    s->mark_input(true);
    d->mark_output(true);
    tg::copy(s, d);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for(Index i = 0; i < nelems; ++i) { sv[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 0.2f; }
    runtime.bind_data(s, sv);
    std::vector<float> dv(nelems, 0.f);
    runtime.bind_data(d, dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(d);
    nntile::core::Tile<fp32_t> ts(sh), td(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = ts.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(sv[static_cast<size_t>(i)]); }
        l1.release();
    }
    nntile::core::copy<fp32_t>(-1, ts, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    nntile::test::require_relative_element_error(gout, tref);
}
