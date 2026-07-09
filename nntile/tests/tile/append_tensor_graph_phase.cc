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
#include "nntile/graph.hh"

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
    TensorGraph::TensorNode *a = tg.data({3}, DataType::FP32)->set_name("a");
    TensorGraph::TensorNode *b = tg.data({3}, DataType::FP32)->set_name("b");
    a->mark_input(true);
    b->mark_input(true);
    TensorGraph::TensorNode *c = gt::add(1.0f, a, 1.0f, b)->set_name("c");

    TensorGraph::PhaseSnapshot p1 = tg.seal_phase();
    TensorGraphTiling til1 = TensorGraphTiling::from_tensor_graph(tg);

    TileGraph tile("tile_inc");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    append_tensor_graph_phase(tg, p1, til1, tile, st, tm);

    REQUIRE(c != nullptr);
    TensorGraph::TensorNode *d = gt::add(1.0f, c, 1.0f, a)->set_name("d");
    REQUIRE(d != nullptr);
    d->mark_output(true);

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
    "Runtime compile incremental",
    "[graph][tile]")
{
    TensorGraph tg("rt_inc");
    TensorGraph::TensorNode *x = tg.data({2}, DataType::FP32)->set_name("x");
    x->mark_input(true);
    TensorGraph::TensorNode *y = gt::scale(2.0f, x)->set_name("y");
    // Carry y across phases: must stay marked or compile reclaim frees it
    // before a later full execute() can rewrite it.
    y->mark_output(true);
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
    TensorGraph::TensorNode *z = gt::add(1.0f, y, 1.0f, x)->set_name("z");
    z->mark_output(true);
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
    TensorGraph::TensorNode *x = tg.data({2}, DataType::FP32)->set_name("x");
    x->mark_input(true);
    TensorGraph::TensorNode *y = gt::scale(2.0f, x)->set_name("y");
    TensorGraph::TensorNode *z = gt::add(1.0f, y, 1.0f, x)->set_name("z");
    z->mark_output(true);

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
    "Runtime compile invalidates tiles after mark_output false",
    "[graph][tile]")
{
    TensorGraph tg("reclaim");
    TensorGraph::TensorNode *x = tg.data({4}, DataType::FP32)->set_name("x");
    x->mark_input(true);
    // Two outputs so clearing one does not hit the no-output DCE fallback.
    TensorGraph::TensorNode *y = gt::scale(2.0f, x)->set_name("y");
    y->mark_output(true);
    TensorGraph::TensorNode *z = gt::scale(3.0f, x)->set_name("z");
    z->mark_output(true);

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

    // Drop output mark on y only: next compile must invalidate y's buffer.
    y->mark_output(false);
    rt.compile();

    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>>
        after;
    rt.export_all_tiles(after);
    REQUIRE(after.count(y) == 0);
    REQUIRE(after.count(z) == 1);
    REQUIRE(after.count(x) == 1);

    // Re-mark y and use it in a new phase; buffer is reallocated.
    y->mark_output(true);
    TensorGraph::TensorNode *w = gt::add(1.0f, y, 1.0f, x)->set_name("w");
    w->mark_output(true);
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
    TensorGraph::TensorNode *a = tg.data({6}, DataType::FP32)->set_name("a");
    TensorGraph::TensorNode *b = tg.data({6}, DataType::FP32)->set_name("b");
    a->mark_input(true);
    b->mark_input(true);
    TensorGraph::TensorNode *c = gt::add(1.0f, a, 1.0f, b)->set_name("c");

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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "compile_incremental_nn_phase pushes tensor_phase_archive",
    "[graph][tile]")
{
    NNGraph nn("phase_arch");
    TensorGraph &tg = nn.tensor_graph();
    TensorGraph::TensorNode *a = tg.data({3}, DataType::FP32)->set_name("a");
    TensorGraph::TensorNode *b = tg.data({3}, DataType::FP32)->set_name("b");
    a->mark_input(true);
    b->mark_input(true);
    gt::add(1.0f, a, 1.0f, b)->set_name("c");

    FinishedTensorPhase fp = nn.finish_phase(false);
    REQUIRE(fp.tensor_graph == &tg);
    REQUIRE(nn.tensor_phase_archives().empty());

    TileGraph tile("tile_phase_arch");
    TileGraphIncrementalState st;
    TensorNodeToTileMap tm;
    Runtime rt(tile);
    compile_incremental_nn_phase(fp,
        nn,
        TensorGraphTiling::from_tensor_graph(tg),
        tile,
        rt,
        st,
        tm,
        true);
    REQUIRE(nn.tensor_phase_archives().size() == 1);
    REQUIRE(nn.tensor_phase_archives()[0].tile_op_end >
            nn.tensor_phase_archives()[0].tile_op_begin);
}
