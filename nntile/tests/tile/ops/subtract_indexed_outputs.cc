/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile_graph/subtract_indexed_outputs.cc
 * Test TileGraph subtract indexed outputs vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "tile_graph_shape_helpers.hh"
#include "nntile/tile/ops/subtract_indexed_outputs.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/subtract_indexed_outputs.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
using namespace nntile::test::tile_graph_shapes;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph subtract_indexed_outputs", "[graph][tile]")
{
    const std::vector<Index> stor_lh = {2, 2};
    const std::vector<Index> graph_lh = graph_shape(stor_lh);
    const std::vector<Index> stor_dh = {3, 2, 2};
    const std::vector<Index> graph_dh = graph_shape(stor_dh);
    const Index nl = 4, nd = 3 * 2 * 2;
    const Scalar v = 0.5;
    const Index ign = -1;
    TileGraph g("g");
    auto* lab = g.data(graph_lh, "labels", DataType::INT64);
    auto* d = g.data(graph_dh, "d", DataType::FP32);
    lab->mark_input(true);
    d->mark_input(true);
    d->mark_output(true);
    tg::subtract_indexed_outputs(v, lab, d, ign);
    Runtime r(g);
    r.compile();
    std::vector<std::int64_t> lv(4);
    lv[0] = 0;
    lv[1] = 1;
    lv[2] = 2;
    lv[3] = 0;
    std::vector<float> dv(nd);
    for(Index i = 0; i < nd; ++i)
    {
        dv[static_cast<size_t>(i)] = 1.0f + 0.1f * static_cast<float>(i);
    }
    r.bind_data(lab, lv);
    r.bind_data(d, dv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<nntile::int64_t> L(stor_lh);
    nntile::core::Tile<fp32_t> D(stor_dh);
    using Y = typename fp32_t::repr_t;
    {
        auto a = L.acquire(STARPU_W);
        a[0] = 0;
        a[1] = 1;
        a[2] = 2;
        a[3] = 0;
        a.release();
    }
    {
        auto b = D.acquire(STARPU_W);
        for(Index i = 0; i < nd; ++i)
        {
            b[i] = Y(1.0f + 0.1f * static_cast<float>(i));
        }
        b.release();
    }
    nntile::core::subtract_indexed_outputs<fp32_t>(-1, v, L, D, ign);
    starpu_task_wait_for_all();
    std::vector<float> tr(nd);
    {
        auto L2 = D.acquire(STARPU_R);
        for(Index i = 0; i < nd; ++i)
        {
            tr[static_cast<size_t>(i)] = static_cast<float>(L2[i]);
        }
        L2.release();
    }
    for(size_t i = 0; i < nd; ++i)
    {
        REQUIRE(std::abs(gout[i] - tr[i]) < 1e-4f);
    }
}
