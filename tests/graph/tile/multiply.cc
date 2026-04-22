/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/multiply.cc
 * multiply: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

#include "context_fixture.hh"
#include "mixed_tile_common.hh"
#include <nntile/graph.hh>

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace tt = nntile::graph::tile_tests;

TEST_CASE("multiply mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    TensorGraph g_ref("ref");
    TensorGraph::TensorNode* a = g_ref.data({10, 12}, "a", DataType::FP32);
    TensorGraph::TensorNode* b = g_ref.data({10, 12}, "b", DataType::FP32);
    a->mark_input(true);
    b->mark_input(true);
    TensorGraph::TensorNode* z =
        gt::multiply(a, b, "z", Scalar{0.5f});
    z->mark_output(true);

    TensorGraph g_tile("tile");
    TensorGraph::TensorNode* at = g_tile.data({10, 12}, "a", DataType::FP32);
    TensorGraph::TensorNode* bt = g_tile.data({10, 12}, "b", DataType::FP32);
    at->mark_input(true);
    bt->mark_input(true);
    tt::apply_mixed_tile_sizes_2d(at);
    tt::apply_mixed_tile_sizes_2d(bt);
    TensorGraph::TensorNode* zt =
        gt::multiply(at, bt, "z", Scalar{0.5f});
    zt->mark_output(true);

    std::vector<float> ad(10 * 12), bd(10 * 12);
    std::mt19937 gen(2);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for(size_t i = 0; i < ad.size(); ++i)
    {
        ad[i] = u(gen);
        bd[i] = u(gen);
    }

    TensorGraph::Runtime rt_ref(g_ref);
    rt_ref.compile();
    rt_ref.bind_data("a", ad);
    rt_ref.bind_data("b", bd);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> y_ref = rt_ref.get_output<float>("z");

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("a", ad);
    rt_tile.bind_data("b", bd);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> y_tile = rt_tile.get_output<float>("z");

    REQUIRE(tt::max_rel_err(y_ref, y_tile) < 5e-4f);
    REQUIRE(tt::frob_rel_err(y_ref, y_tile) < 5e-4f);
}
