/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/hypot_scalar_inverse.cc
 * Test TileGraph hypot scalar inverse vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "context_fixture.hh"
#include "nntile/graph/tile/hypot_scalar_inverse.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/hypot_scalar_inverse.hh"
#include "nntile/tile/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph hypot_scalar_inverse matches tile", "[graph][tile]")
{
    const std::vector<Index> sh = {2, 3};
    const Index nelems = 6;
    const Scalar eps = 0.1, alpha = 2.0;
    TileGraph g("g");
    auto* d = g.data(sh, "d", DataType::FP32);
    d->mark_input(true);
    d->mark_output(true);
    tg::hypot_scalar_inverse(eps, alpha, d);
    TileGraph::Runtime runtime(g);
    runtime.compile();
    std::vector<float> dv(nelems);
    for(Index i = 0; i < nelems; ++i) { dv[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i) + 0.5f; }
    runtime.bind_data("d", dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>("d");
    nntile::tile::Tile<fp32_t> td(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(dv[static_cast<size_t>(i)]); }
        l1.release();
    }
    nntile::tile::hypot_scalar_inverse<fp32_t>(eps, alpha, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-3f;
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
