/*! @copyright (c) 2022-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile_graph/append_tensor_graph_phase.cc
 * Tests incremental TensorGraph phase lowering.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/append_tensor_graph_phase.hh"

#include "context_fixture.hh"
#include "nntile/tensor/ops/fill.hh"
#include <nntile/defs.h>
#include <nntile/tensor.hh>
#include <nntile/tile.hh>
#include <nntile/runtime.hh>
#include <nntile/tile/append_tensor_graph_phase.hh>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "append_tensor_graph_phase two additive phases",
    "[graph][tile]")
{
    TensorGraph tg("inc_add");
    nntile::TensorRef a = tg.data({3}, DataType::FP32);
        a->set_name("a");
    nntile::TensorRef b = tg.data({3}, DataType::FP32);
        b->set_name("b");
    nntile::TensorRef c = nntile::TensorRef::adopt(gt::add(1.0f, a, 1.0f, b));
    c->set_name("c");

    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TensorGraphTiling til1 = TensorGraphTiling::from_tensor_graph(tg);

    TileGraph tile("tile_inc");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(tg, p1, til1, tile, st, tm);

    REQUIRE(c != nullptr);
    nntile::TensorRef d = nntile::TensorRef::adopt(gt::add(1.0f, c, 1.0f, a));
    d->set_name("d");
    REQUIRE(d != nullptr);

    TensorGraph::PhaseSnapshot p2 = tg.seal_phase();
    TensorGraphTiling til2 = TensorGraphTiling::from_tensor_graph(tg);
    append_tensor_graph_phase(tg, p2, til2, tile, st, tm);

    Runtime rt(tile);
    rt.compile();

    std::vector<float> av = {1.f, 2.f, 3.f};
    std::vector<float> bv = {4.f, 5.f, 6.f};
    rt.bind_data(a, av);
    rt.bind_data(b, bv);
    rt.execute();
    rt.wait();

    std::vector<float> out = rt.get_output<float>(d);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == 6.f);
    REQUIRE(out[1] == 9.f);
    REQUIRE(out[2] == 12.f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "append_tensor_graph_phase factory-only after ensure_phase_layouts",
    "[graph][tile]")
{
    // Mirrors torch_nntile compile_graph: ensure_phase_layouts then
    // append. Per-TU touch_gen counters used to skip every tensor on
    // this first factory-only phase (FILL / arange).
    TensorGraph tg("fill_first");
    nntile::TensorRef x = tg.data({4}, DataType::FP32);
    x->set_name("x");
    gt::fill(Scalar(2.5), x);

    TensorGraph::PhaseSnapshot phase = tg.seal_phase();
    auto tiling = std::make_shared<TensorGraphTiling>();
    tiling->ensure_phase_layouts(tg, phase);

    TileGraph tile("tile_fill_first");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(tg, phase, tiling, tile, st, tm);

    Runtime rt(tile);
    rt.compile();
    rt.execute();
    rt.wait();

    std::vector<float> out = rt.get_output<float>(x);
    REQUIRE(out.size() == 4);
    REQUIRE(out[0] == 2.5f);
    REQUIRE(out[3] == 2.5f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Runtime compile incremental",
    "[graph][tile]")
{
    TensorGraph tg("rt_inc");
    nntile::TensorRef x = tg.data({2}, DataType::FP32);
        x->set_name("x");
    nntile::TensorRef y = nntile::TensorRef::adopt(gt::scale(2.0f, x));
    y->set_name("y");
    // Carry y across phases: must stay marked or compile reclaim frees it
    // before a later full execute() can rewrite it.
    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TileGraph tile("t2");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(
        tg, p1, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);

    Runtime rt(tile);
    rt.compile();
    const size_t n1 = rt.execution_op_count();
    REQUIRE(n1 > 0);

    REQUIRE(y != nullptr);
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::add(1.0f, y, 1.0f, x));
    z->set_name("z");
    TensorGraph::PhaseSnapshot p2 = tg.seal_phase();
    append_tensor_graph_phase(
        tg, p2, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);

    rt.compile();
    REQUIRE(rt.execution_op_count() > n1);

    std::vector<float> xv = {2.f, 3.f};
    rt.bind_data(x, xv);
    rt.execute();
    rt.wait();
    std::vector<float> zout = rt.get_output<float>(z);
    REQUIRE(zout[0] == 6.f);
    REQUIRE(zout[1] == 9.f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Runtime execute reallocates unmarked tiles after recompile",
    "[graph][tile]")
{
    // Unmarked intermediate: freed after first execute / next compile, but
    // a subsequent full execute() must still be able to re-run from op 0.
    TensorGraph tg("reexec");
    nntile::TensorRef x = tg.data({2}, DataType::FP32);
        x->set_name("x");
    nntile::TensorRef y = nntile::TensorRef::adopt(gt::scale(2.0f, x));
    y->set_name("y");
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::add(1.0f, y, 1.0f, x));
    z->set_name("z");

    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TileGraph tile("t_reexec");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(
        tg, p1, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);

    Runtime rt(tile);
    rt.compile();
    rt.bind_data(x, std::vector<float>{2.f, 3.f});
    rt.execute();
    rt.wait();
    std::vector<float> first = rt.get_output<float>(z);
    REQUIRE(first[0] == 6.f);
    REQUIRE(first[1] == 9.f);

    // Recompile with watermark at end: pending slice empty; unmarked y must
    // not stay allocated. Full execute() must still succeed.
    rt.compile();
    rt.execute();
    rt.wait();
    std::vector<float> second = rt.get_output<float>(z);
    REQUIRE(second[0] == 6.f);
    REQUIRE(second[1] == 9.f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Runtime invalidate_logical_tiles after TensorRef release",
    "[graph][tile]")
{
    TensorGraph tg("reclaim");
    nntile::TensorRef x = tg.data({4}, DataType::FP32);
        x->set_name("x");
    // Two outputs so clearing one does not hit the no-output DCE fallback.
    nntile::TensorRef y = nntile::TensorRef::adopt(gt::scale(2.0f, x));
    y->set_name("y");
    nntile::TensorRef z = nntile::TensorRef::adopt(gt::scale(3.0f, x));
    z->set_name("z");

    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TileGraph tile("t_reclaim");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(
        tg, p1, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);

    Runtime rt(tile);
    rt.compile();
    rt.bind_data(x, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    rt.execute();
    rt.wait();
    std::vector<float> y_out = rt.get_output<float>(y);
    REQUIRE(y_out.size() == 4);
    REQUIRE(y_out[0] == 2.f);
    REQUIRE(y_out[1] == 4.f);
    REQUIRE(y_out[2] == 6.f);
    REQUIRE(y_out[3] == 8.f);

    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>>
        before;
    rt.export_all_tiles(before);
    REQUIRE(before.count(y) == 1);
    REQUIRE(before.count(z) == 1);

    // While live, reclaim is a no-op.
    rt.invalidate_logical_tiles(y.get());
    {
        std::unordered_map<TensorGraph::TensorNode const *,
            std::vector<std::shared_ptr<void>>>
            still;
        rt.export_all_tiles(still);
        REQUIRE(still.count(y) == 1);
    }

    // Release without recording graph INVALIDATE (re-adopt below).
    TensorGraph::TensorNode *y_raw = y.get();
    nntile::set_tensor_nodes_alive(false);
    y = nntile::TensorRef{};
    nntile::set_tensor_nodes_alive(true);
    rt.invalidate_logical_tiles(y_raw);

    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>>
        after;
    rt.export_all_tiles(after);
    REQUIRE(after.count(y_raw) == 0);
    REQUIRE(after.count(z) == 1);
    REQUIRE(after.count(x) == 1);

    y = nntile::TensorRef::adopt(y_raw);
    nntile::TensorRef w = nntile::TensorRef::adopt(gt::add(1.0f, y, 1.0f, x));
    w->set_name("w");
    TensorGraph::PhaseSnapshot p2 = tg.seal_phase();
    append_tensor_graph_phase(
        tg, p2, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);
    rt.compile();
    rt.execute();
    rt.wait();
    std::vector<float> w_out = rt.get_output<float>(w);
    REQUIRE(w_out.size() == 4);
    REQUIRE(w_out[0] == 3.f);
    REQUIRE(w_out[1] == 6.f);
    REQUIRE(w_out[2] == 9.f);
    REQUIRE(w_out[3] == 12.f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "append_tensor_graph_phase throws on tiling change",
    "[graph][tile]")
{
    TensorGraph tg("tiling_change");
    nntile::TensorRef a = tg.data({6}, DataType::FP32);
        a->set_name("a");
    nntile::TensorRef b = tg.data({6}, DataType::FP32);
        b->set_name("b");
    nntile::TensorRef c = nntile::TensorRef::adopt(gt::add(1.0f, a, 1.0f, b));
    c->set_name("c");

    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TileGraph tile("tile_tc");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(
        tg, p1, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm);

    a->mutable_axes()[0]->set_tiling(std::vector<Index>{3, 3});

    REQUIRE(c != nullptr);
    gt::add(1.0f, c, 1.0f, a)->set_name("d");
    TensorGraph::PhaseSnapshot p2 = tg.seal_phase();

    REQUIRE_THROWS_AS(
        append_tensor_graph_phase(
            tg, p2, TensorGraphTiling::from_tensor_graph(tg), tile, st, tm),
        std::runtime_error);
}

