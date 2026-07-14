/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/pow.cc
 * pow (in-place): TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "mixed_tile_common.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/graph.hh>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;
namespace tt = nntile::core_tests;

TEST_CASE("pow mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    TensorGraph g_ref("ref");
    nntile::TensorRef x_ref = g_ref.data({10, 12}, DataType::FP32);
        x_ref->set_name("x");
    gt::pow(Scalar{0.5f}, Scalar{2.f}, x_ref);

    TensorGraph g_tile("tile");
    nntile::TensorRef x_tile = g_tile.data({10, 12}, DataType::FP32);
        x_tile->set_name("x");
    tt::apply_mixed_tile_sizes_2d(x_tile);
    gt::pow(Scalar{0.5f}, Scalar{2.f}, x_tile);

    std::vector<float> x_data(10 * 12, 0.25f);

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);

    Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data(x_ref, x_data);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> out_ref = rt_ref.get_output<float>(x_ref);

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data(x_tile, x_data);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> out_tile = rt_tile.get_output<float>(x_tile);

    nntile::test::require_relative_element_error(out_ref, out_tile);
}
