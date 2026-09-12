#include <nntile/tensor/tensor_ref.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile/ddp_rewrite.cc
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"

#include <nntile/context.hh>
#include <nntile/tensor.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/ops/fill.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tile.hh>
#include <nntile/runtime.hh>
#include <nntile/tile/ddp.hh>

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace nntile;
namespace gt = nntile::tensor;
namespace tl = nntile::tile;

namespace
{

struct TwoCpuContextFixture
{
    Context context;

    TwoCpuContextFixture()
        : context(2, 0, 0, "/tmp/nntile_ooc_ddp", 16777216, 0)
    {
    }
};

std::vector<float> sequential_range(size_t n)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = static_cast<float>(i + 1) * 0.01f;
    }
    return v;
}

} // namespace

TEST_CASE("ddp_tile_sizes even split", "[ddp]")
{
    REQUIRE(tl::ddp_tile_sizes(4, 2) == std::vector<Index>{2, 2});
    REQUIRE(tl::ddp_tile_sizes(5, 2) == std::vector<Index>{3, 2});
    REQUIRE(tl::ddp_tile_sizes(4, 1) == std::vector<Index>{4});
    REQUIRE_THROWS_AS(tl::ddp_tile_sizes(2, 4), std::invalid_argument);
}

TEST_CASE("enable_ddp rejects a second axis", "[ddp]")
{
    TensorGraph graph("ddp_axis");
    graph.enable_ddp("batch");
    REQUIRE(graph.ddp_enabled());
    REQUIRE(graph.ddp_axis() == "batch");
    graph.enable_ddp("batch");
    REQUIRE_THROWS_AS(graph.enable_ddp("hidden"), std::runtime_error);
}

TEST_CASE_METHOD(TwoCpuContextFixture,
    "ddp rewrite keeps one W tile and adds dW locals",
    "[ddp]")
{
    TensorGraph graph("ddp_gemm");
    nntile::TensorRef a = graph.data({4, 8}, DataType::FP32);
    a->set_name("A");
    a->axis(0)->name = "batch";
    nntile::TensorRef w = graph.data({8, 6}, DataType::FP32);
    w->set_name("W");
    nntile::TensorRef c = nntile::TensorRef::adopt(
        gt::gemm(a, w, 1.0, false, false, 1, 0));
    c->set_name("C");
    nntile::TensorRef dc = graph.data({4, 6}, DataType::FP32);
    dc->set_name("dC");
    merge_axis(c->mutable_axes()[0], dc->mutable_axes()[0]);
    nntile::TensorRef dw = graph.data({8, 6}, DataType::FP32);
    dw->set_name("dW");
    gt::fill(0.0, dw);
    gt::gemm(a, dc, dw, 1.0, 0.0, true, false, 1, 0);
    nntile::TensorRef vel = graph.data({8, 6}, DataType::FP32);
    vel->set_name("V");
    gt::fill(0.0, vel);
    gt::sgd_step(1, 0.0, 0.1, 0.0, 0.0, false, dw, vel, w);

    graph.enable_ddp("batch");
    tl::apply_ddp_axis_tiling(graph, graph.ddp_axis());

    AxisDescriptor const *batch = a->axis(0);
    REQUIRE(batch->is_tiled());
    REQUIRE(batch->tile_sizes == std::vector<Index>{2, 2});
    REQUIRE_FALSE(w->axis(0)->is_tiled());

    TileGraph tiles = TileGraph::from_tensor_graph(graph);
    tl::rewrite_ddp_pending(tiles, 0, tiles.num_ops(), "batch");

    TileGraph::TensorDescriptor const *a_desc =
        tiles.get_tensor_descriptor(a);
    TileGraph::TensorDescriptor const *w_desc =
        tiles.get_tensor_descriptor(w);
    TileGraph::TensorDescriptor const *dw_desc =
        tiles.get_tensor_descriptor(dw);
    REQUIRE(a_desc != nullptr);
    REQUIRE(w_desc != nullptr);
    REQUIRE(dw_desc != nullptr);
    REQUIRE(a_desc->tiles.size() == 2);
    REQUIRE(w_desc->tiles.size() == 1);
    REQUIRE(dw_desc->tiles.size() == 1);

    int extra_dw = 0;
    int add_ops = 0;
    int hinted_gemm = 0;
    bool saw_sgd_after_add = false;
    bool saw_add = false;
    for (auto const &op : tiles.ops())
    {
        if (op->op_name() == "TILE_ADD_INPLACE")
        {
            ++add_ops;
            saw_add = true;
        }
        if (op->op_name() == "TILE_SGD_STEP" && saw_add)
        {
            saw_sgd_after_add = true;
        }
        if (op->op_name() == "TILE_GEMM")
        {
            if (op->device_hint() >= 0)
            {
                ++hinted_gemm;
            }
        }
        for (TileGraph::TileNode *t : op->outputs())
        {
            if (t == nullptr || t->tensor_descriptor() != nullptr)
            {
                continue;
            }
            if (t->shape() == dw->shape())
            {
                ++extra_dw;
            }
        }
    }
    REQUIRE(add_ops == 2);
    REQUIRE(extra_dw >= 2);
    REQUIRE(hinted_gemm >= 1);
    REQUIRE(saw_sgd_after_add);

    std::vector<float> a_data = sequential_range(4 * 8);
    std::vector<float> w_data = sequential_range(8 * 6);
    std::vector<float> dc_data(4 * 6, 1.0f);
    std::vector<float> dw_ref(8 * 6, 0.0f);
    for (Index b = 0; b < 4; ++b)
    {
        for (Index k = 0; k < 8; ++k)
        {
            for (Index n = 0; n < 6; ++n)
            {
                dw_ref[static_cast<size_t>(k * 6 + n)] +=
                    a_data[static_cast<size_t>(b * 8 + k)] *
                    dc_data[static_cast<size_t>(b * 6 + n)];
            }
        }
    }
    std::vector<float> w_ref = w_data;
    for (size_t i = 0; i < w_ref.size(); ++i)
    {
        w_ref[i] -= 0.1f * dw_ref[i];
    }

    Runtime rt(tiles);
    rt.compile();
    rt.bind_data(a, a_data);
    rt.bind_data(w, w_data);
    rt.bind_data(dc, dc_data);
    rt.execute();
    rt.wait();
    std::vector<float> dw_out = rt.get_output<float>(dw);
    std::vector<float> w_out = rt.get_output<float>(w);
    REQUIRE(dw_out.size() == dw_ref.size());
    for (size_t i = 0; i < dw_ref.size(); ++i)
    {
        REQUIRE(std::fabs(dw_out[i] - dw_ref[i]) < 1e-4f);
    }
    for (size_t i = 0; i < w_ref.size(); ++i)
    {
        REQUIRE(std::fabs(w_out[i] - w_ref[i]) < 1e-4f);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "ddp rewrite is a no-op extra-tile-wise when N=1",
    "[ddp]")
{
    TensorGraph graph("ddp_n1");
    nntile::TensorRef a = graph.data({4, 8}, DataType::FP32);
    a->axis(0)->name = "batch";
    nntile::TensorRef w = graph.data({8, 6}, DataType::FP32);
    nntile::TensorRef c = nntile::TensorRef::adopt(
        gt::gemm(a, w, 1.0, false, false, 1, 0));
    nntile::TensorRef dc = graph.data({4, 6}, DataType::FP32);
    merge_axis(c->mutable_axes()[0], dc->mutable_axes()[0]);
    nntile::TensorRef dw = graph.data({8, 6}, DataType::FP32);
    gt::fill(0.0, dw);
    gt::gemm(a, dc, dw, 1.0, 0.0, true, false, 1, 0);

    graph.enable_ddp("batch");
    tl::apply_ddp_axis_tiling(graph, graph.ddp_axis());
    REQUIRE_FALSE(a->axis(0)->is_tiled());

    TileGraph tiles = TileGraph::from_tensor_graph(graph);
    size_t const nops = tiles.num_ops();
    tl::rewrite_ddp_pending(tiles, 0, tiles.num_ops(), "batch");
    REQUIRE(tiles.num_ops() == nops);
    int adds = 0;
    for (auto const &op : tiles.ops())
    {
        if (op->op_name() == "TILE_ADD_INPLACE")
        {
            ++adds;
        }
    }
    REQUIRE(adds == 0);
}
