/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/log_scalar.cc
 * Test TileGraph log_scalar (names a StarPU data handle).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"
#include "nntile/graph/tile/log_scalar.hh"
#include "nntile/graph/tile.hh"

using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;

TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph log_scalar runs", "[graph][tile]")
{
    TileGraph g("g");
    auto* v = g.data(std::vector<Index>{1}, "v", DataType::FP32);
    v->mark_input(true);
    v->mark_output(true);
    tg::log_scalar("test", v);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<float> x(1, 1.f);
    r.bind_data("v", x);
    r.execute();
    r.wait();
    REQUIRE(r.get_output<float>("v").size() == 1u);
}
