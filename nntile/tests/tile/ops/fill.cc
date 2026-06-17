/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/fill.cc
 * Test TileGraph fill vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/fill.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/fill.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph fill matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 2};
    const Index nelems = 6;
    const Scalar v = 3.25;
    TileGraph g("g");
    auto* x = g.data(sh, "x", DataType::FP32);
    x->mark_input(true);
    x->mark_output(true);
    tg::fill(v, x);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> xv(nelems, 0.f);
    runtime.bind_data(x, xv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(x);
    nntile::core::Tile<fp32_t> tx(sh);
    nntile::core::fill<fp32_t>(-1, v, tx);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = tx.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    nntile::test::require_relative_element_error(gout, tref);
}
