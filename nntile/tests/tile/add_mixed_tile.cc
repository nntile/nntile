/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/add_mixed_tile.cc
 * add: TensorGraph vs TileGraph (mixed tile sizes) parity.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "mixed_tile_common.hh"
#include "test_frobenius.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/graph.hh>
#include <random>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;
namespace tt = nntile::core_tests;

TEST_CASE("add mixed tile parity", "[graph][tile]")
{
    test::ContextFixture fx;

    TensorGraph g_ref("ref");
    nntile::TensorRef a = g_ref.data({10, 12}, DataType::FP32);
        a->set_name("a");
    nntile::TensorRef b = g_ref.data({10, 12}, DataType::FP32);
        b->set_name("b");
    nntile::TensorRef out = nntile::TensorRef::adopt(gt::add(Scalar{1.f}, a, Scalar{1.f}, b));
    out->set_name("out");

    TensorGraph g_tile("tile");
    nntile::TensorRef at = g_tile.data({10, 12}, DataType::FP32);
        at->set_name("a");
    nntile::TensorRef bt = g_tile.data({10, 12}, DataType::FP32);
        bt->set_name("b");
    tt::apply_mixed_tile_sizes_2d(at);
    tt::apply_mixed_tile_sizes_2d(bt);
    nntile::TensorRef outt = nntile::TensorRef::adopt(gt::add(Scalar{1.f}, at, Scalar{1.f}, bt));
    outt->set_name("out");

    std::vector<float> ad(10 * 12), bd(10 * 12);
    std::mt19937 gen(2);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (size_t i = 0; i < ad.size(); ++i)
    {
        ad[i] = u(gen);
        bd[i] = u(gen);
    }

    TileGraph rt_ref_tile = TileGraph::from_tensor_graph(g_ref);

    Runtime rt_ref(rt_ref_tile);
    rt_ref.compile();
    rt_ref.bind_data(a, ad);
    rt_ref.bind_data(b, bd);
    rt_ref.execute();
    rt_ref.wait();
    const std::vector<float> y_ref = rt_ref.get_output<float>(out);

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile);
    Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data(at, ad);
    rt_tile.bind_data(bt, bd);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> y_tile = rt_tile.get_output<float>(outt);

    nntile::test::require_relative_element_error(y_tile, y_ref);
}
