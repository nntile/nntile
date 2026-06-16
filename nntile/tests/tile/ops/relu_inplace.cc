/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/relu_inplace.cc
 * TileGraph relu_inplace vs nntile::core::relu_inplace (in-place parity B).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/relu_inplace.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/relu_inplace.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph relu_inplace matches tile", "[graph][tile]")
{
    const std::vector<Index> stor_sh = {2, 3};
    const std::vector<Index> graph_sh = graph_shape(stor_sh);
    const Index nelems = 6;
    TileGraph g("g");
    auto* d = g.data(graph_sh, "d", DataType::FP32);
    d->mark_input(true);
    d->mark_output(true);
    tg::relu_inplace(d);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> dv(nelems);
    for(Index i = 0; i < nelems; ++i) { dv[static_cast<size_t>(i)] = static_cast<float>(i) * 0.2f - 0.3f; }
    runtime.bind_data(d, dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(d);
    nntile::core::Tile<fp32_t> td(stor_sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = td.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i) { l1[i] = Y(dv[static_cast<size_t>(i)]); }
        l1.release();
    }
    nntile::core::relu_inplace<fp32_t>(-1, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    constexpr float tol = 1e-4f;
    REQUIRE(gout.size() == tref.size());
    for(size_t i = 0; i < tref.size(); ++i) { REQUIRE(std::abs(gout[i] - tref[i]) < tol); }
}
