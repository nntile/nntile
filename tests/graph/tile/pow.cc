/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/pow.cc
 * pow (in-place): TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "context_fixture.hh"
#include "mixed_tile_common.hh"
#include <nntile/graph.hh>

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace tt = nntile::graph::tile_tests;

TEST_CASE("pow mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    TensorGraph g_ref("ref");
    TensorGraph::TensorNode* x_ref =
        g_ref.data({10, 12}, "x", DataType::FP32);
    x_ref->mark_input(true);
    gt::pow(Scalar{0.5f}, Scalar{2.f}, x_ref);
    x_ref->mark_output(true);

    TensorGraph g_tile("tile");
    TensorGraph::TensorNode* x_tile =
        g_tile.data({10, 12}, "x", DataType::FP32);
    x_tile->mark_input(true);
    tt::apply_mixed_tile_sizes_2d(x_tile);
    gt::pow(Scalar{0.5f}, Scalar{2.f}, x_tile);
    x_tile->mark_output(true);

    std::vector<float> x_data(10 * 12, 0.25f);

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);


    TileGraph::Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data("x", x_data);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> out_ref = rt_ref.get_output<float>("x");

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("x", x_data);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> out_tile = rt_tile.get_output<float>("x");

    REQUIRE(tt::max_rel_err(out_ref, out_tile) < 5e-4f);
    REQUIRE(tt::frob_rel_err(out_ref, out_tile) < 5e-4f);
}
