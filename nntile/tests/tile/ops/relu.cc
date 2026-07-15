/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/relu.cc
 * TileGraph relu vs nntile::core::relu (small parity B).
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/relu.hh"

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "mixed_tile_common.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/relu.hh"
#include "nntile/core/tile.hh"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <nntile/graph.hh>
#include <numeric>
#include <random>
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
namespace gt = nntile::tensor;
namespace tt = nntile::core_tests;
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph relu matches tile",
    "[graph][tile]")
{
    const std::vector<Index> sh = {3, 2};
    const Index nelems = 6;
    TileGraph g("g");
    auto *s = g.data(sh, "s", DataType::FP32);
    auto *d = g.data(sh, "d", DataType::FP32);
    tg::relu(s, d);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        sv[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 0.2f;
    }
    runtime.bind_data(s, sv);
    std::vector<float> dv(nelems, 0.f);
    runtime.bind_data(d, dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(d);
    nntile::core::Tile<fp32_t> ts(sh), td(sh);
    using Y = typename nntile::fp32_t::repr_t;
    {
        auto l1 = ts.acquire(STARPU_W);
        for (Index i = 0; i < nelems; ++i)
        {
            l1[i] = Y(sv[static_cast<size_t>(i)]);
        }
        l1.release();
    }
    nntile::core::relu<fp32_t>(-1, ts, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(nelems);
    {
        auto l2 = td.acquire(STARPU_R);
        for (Index i = 0; i < nelems; ++i)
        {
            tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]);
        }
        l2.release();
    }
    nntile::test::require_relative_element_error(gout, tref);
}

TEST_CASE("ReLU mixed tile parity (TensorGraph ref vs TileGraph tile)",
    "[graph][tile]")
{
    test::ContextFixture fx;
    (void) fx;
    TensorGraph g_ref("ref");
    nntile::TensorRef x_ref = g_ref.data({10, 12}, DataType::FP32);
    x_ref->set_name("x");
    nntile::TensorRef y_ref_node = nntile::TensorRef::adopt(gt::relu(x_ref));

    y_ref_node->set_name("y");

    TensorGraph g_tile("tile");
    nntile::TensorRef x_tile = g_tile.data({10, 12}, DataType::FP32);
    x_tile->set_name("x");
    tt::apply_mixed_tile_sizes_2d(x_tile);
    nntile::TensorRef y_tile_node = nntile::TensorRef::adopt(gt::relu(x_tile));

    y_tile_node->set_name("y");

    std::mt19937 gen(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
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
    const std::vector<float> y_out_ref = rt_ref.get_output<float>(y_ref_node);

    TileGraph tgraph = TileGraph::from_tensor_graph(g_tile);
    Runtime rt_tile(tgraph);
    rt_tile.compile();
    rt_tile.bind_data(x_tile, x_data);
    rt_tile.execute();
    rt_tile.wait();
    const std::vector<float> y_out_tile =
        rt_tile.get_output<float>(y_tile_node);

    nntile::test::require_relative_element_error(y_out_ref, y_out_tile);
}
