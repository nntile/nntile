/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/gelu.cc
 * GeLU: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "mixed_tile_common.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/graph.hh>
#include <random>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;
namespace tt = nntile::core_tests;

TEST_CASE("GeLU mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    TensorGraph g_ref("ref");
    nntile::TensorRef x_ref = g_ref.data({10, 12}, DataType::FP32);
    x_ref->set_name("x");
    nntile::TensorRef y_ref_node = nntile::TensorRef::adopt(gt::gelu(x_ref));

    y_ref_node->set_name("y");

    TensorGraph g_tile("tile");
    nntile::TensorRef x_tile = g_tile.data({10, 12}, DataType::FP32);
    x_tile->set_name("x");
    tt::apply_mixed_tile_sizes_2d(x_tile);
    nntile::TensorRef y_tile_node = nntile::TensorRef::adopt(gt::gelu(x_tile));

    y_tile_node->set_name("y");

    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.f, 0.5f);
    std::vector<float> x_data(10 * 12);
    for (auto &v : x_data)
    {
        v = dist(gen);
    }

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);

    Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data(x_ref, x_data);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> y_ref = rt_ref.get_output<float>(y_ref_node);

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data(x_tile, x_data);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> y_tile = rt_tile.get_output<float>(y_tile_node);

    nntile::test::require_relative_element_error(y_ref, y_tile);
}
