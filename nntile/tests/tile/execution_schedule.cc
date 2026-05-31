/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile/execution_schedule.cc
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nntile/context.hh>
#include <nntile/core/execution_schedule.hh>
#include <nntile/runtime.hh>
#include <nntile/tile/ops/add.hh>

using namespace nntile;
using namespace nntile::tile;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "op worker matches virtual owner of output tile",
    "[execution_schedule]")
{
    TileGraph tg("sched");
    auto *t0 = tg.data({2}, "t0", DataType::FP32);
    auto *t1 = tg.data({2}, "t1", DataType::FP32);
    auto *x = tg.data({2}, "x", DataType::FP32);

    TileGraph::TensorDescriptor desc;
    desc.tensor_name = "T";
    desc.tensor_shape = {4};
    desc.tile_shape = {2, 2};
    desc.grid_shape = {2};
    desc.dtype = DataType::FP32;
    desc.tiles = {t0, t1};
    tg.add_tensor_descriptor(std::move(desc));

    std::vector<std::shared_ptr<TileGraph::OpNode>> order;
    order.push_back(std::make_shared<TileAddOp>(t0, x, t1, 1.0, 0.0));

    ExecutionSchedule sched = generate_round_robin_execution_schedule(tg, order);
    REQUIRE(sched.num_workers >= 1);
    REQUIRE(sched.ops.size() == 1);
    if (sched.num_workers >= 2)
    {
        REQUIRE(sched.tile_virtual_worker.at("t1") == 1);
        REQUIRE(sched.ops[0].worker == sched.tile_virtual_worker.at("t1"));
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "multi-writable op picks worker with smaller writable volume",
    "[execution_schedule]")
{
    TileGraph tg("multi");
    auto *a0 = tg.data({2}, "a0", DataType::FP32);
    auto *a1 = tg.data({8}, "a1", DataType::FP32);
    auto *b0 = tg.data({2}, "b0", DataType::FP32);
    auto *b1 = tg.data({8}, "b1", DataType::FP32);

    TileGraph::TensorDescriptor da;
    da.tensor_name = "A";
    da.tensor_shape = {10};
    da.tile_shape = {2, 8};
    da.grid_shape = {2};
    da.dtype = DataType::FP32;
    da.tiles = {a0, a1};
    tg.add_tensor_descriptor(std::move(da));

    TileGraph::TensorDescriptor db;
    db.tensor_name = "B";
    db.tensor_shape = {10};
    db.tile_shape = {2, 8};
    db.grid_shape = {2};
    db.dtype = DataType::FP32;
    db.tiles = {b0, b1};
    tg.add_tensor_descriptor(std::move(db));

    std::vector<std::shared_ptr<TileGraph::OpNode>> order;
    order.push_back(std::make_shared<TileAddOp>(a0, b0, a0, 1.0, 0.0));
    order.push_back(std::make_shared<TileAddOp>(a1, b1, a1, 1.0, 0.0));

    ExecutionSchedule sched = generate_round_robin_execution_schedule(tg, order);
    REQUIRE(sched.num_workers >= 1);
    if (sched.num_workers >= 2)
    {
        REQUIRE(sched.ops[0].worker == 0);
        REQUIRE(sched.ops[1].worker == 1);
    }
}

TEST_CASE("execution schedule json export", "[execution_schedule]")
{
    TileGraph tg("json");
    auto *z = tg.data({4}, "z", DataType::FP32);
    auto *x = tg.data({4}, "x", DataType::FP32);
    auto *y = tg.data({4}, "y", DataType::FP32);

    std::vector<std::shared_ptr<TileGraph::OpNode>> order;
    order.push_back(std::make_shared<TileAddOp>(x, y, z, 1.0, 1.0));

    ExecutionSchedule sched = generate_round_robin_execution_schedule(tg, order);
    std::string const js = execution_schedule_to_json(sched);
    REQUIRE(js.find("\"policy\"") != std::string::npos);
    REQUIRE(js.find("\"virtual_tile_workers\"") != std::string::npos);
    REQUIRE(js.find("\"ops\"") != std::string::npos);

    char const *const tmp_path = "/tmp/nntile_execution_schedule_test.json";
    write_execution_schedule_json(sched, tmp_path);
    ExecutionSchedule loaded = load_execution_schedule_json(tmp_path);
    REQUIRE(loaded.ops.size() == sched.ops.size());
    REQUIRE(loaded.ops[0].worker == sched.ops[0].worker);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "runtime loads execution.json before execute",
    "[execution_schedule]")
{
    TileGraph tg("rt");
    auto *x = tg.data({4}, "x", DataType::FP32);
    auto *y = tg.data({4}, "y", DataType::FP32);
    auto *z = tg.data({4}, "z", DataType::FP32);
    x->mark_input(true);
    y->mark_input(true);
    add(1.0, x, 1.0, y, z);
    z->mark_output(true);

    Runtime rt(tg);
    rt.compile();
    rt.set_execution_schedule(rt.generate_round_robin_execution_schedule());
    char const *const tmp_path = "/tmp/nntile_runtime_execution_test.json";
    write_execution_schedule_json(rt.execution_schedule(), tmp_path);
    rt.load_execution_schedule(tmp_path);
    std::vector<float> x_data(4, 1.f);
    std::vector<float> y_data(4, 2.f);
    std::vector<float> z_data(4, 0.f);
    rt.bind_data(x, x_data);
    rt.bind_data(y, y_data);
    rt.bind_data(z, z_data);
    rt.execute();
    rt.wait();
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "set_execution_schedule rejects stale fingerprint",
    "[execution_schedule]")
{
    TileGraph tg("stale");
    auto *x = tg.data({4}, "x", DataType::FP32);
    auto *y = tg.data({4}, "y", DataType::FP32);
    auto *z = tg.data({4}, "z", DataType::FP32);

    std::vector<std::shared_ptr<TileGraph::OpNode>> order;
    order.push_back(std::make_shared<TileAddOp>(x, y, z, 1.0, 1.0));

    Runtime rt(tg);
    rt.compile();
    ExecutionSchedule good = rt.generate_round_robin_execution_schedule();

    ExecutionSchedule bad = good;
    bad.fingerprint.op_count += 1;
    bad.fingerprint.op_names.push_back("TILE_FAKE");

    REQUIRE_THROWS_AS(rt.set_execution_schedule(std::move(bad)),
        std::runtime_error);
}

TEST_CASE("load_execution_schedule_json rejects out-of-range worker",
    "[execution_schedule]")
{
    char const *const tmp_path =
        "/tmp/nntile_execution_schedule_bad_worker.json";
    std::ofstream out(tmp_path);
    out << R"({
  "version": 1,
  "policy": "round_robin_virtual_tensor_split",
  "hardware": {"num_workers": 2, "worker_kind": "cpu"},
  "schedule_fingerprint": {"op_count": 1, "op_names": ["add"]},
  "virtual_tile_workers": [],
  "ops": [{"index": 0, "op": "add", "worker": 9}]
})";
    out.close();
    REQUIRE_THROWS_AS(load_execution_schedule_json(tmp_path),
        std::runtime_error);
}

TEST_CASE("affinity_batch schedule policy and tile map", "[execution_schedule]")
{
    TileGraph tg("batch_aff");
    auto *t0 = tg.data({2}, "t0", DataType::FP32);
    auto *t1 = tg.data({2}, "t1", DataType::FP32);
    auto *t2 = tg.data({2}, "t2", DataType::FP32);
    auto *t3 = tg.data({2}, "t3", DataType::FP32);

    TileGraph::TensorDescriptor da;
    da.tensor_name = "A";
    da.tensor_shape = {4, 4};
    da.tile_shape = {2, 2};
    da.grid_shape = {2, 2};
    da.dtype = DataType::FP32;
    da.tiles = {t0, t1, t2, t3};
    tg.add_tensor_descriptor(std::move(da));

    std::vector<std::shared_ptr<TileGraph::OpNode>> order;
    order.push_back(std::make_shared<TileAddOp>(t0, t1, t0, 1.0, 0.0));

    ExecutionSchedule rr =
        generate_round_robin_execution_schedule(tg, order);
    ExecutionSchedule ab =
        generate_affinity_batch_execution_schedule(tg, order);
    REQUIRE(rr.policy == "round_robin_virtual_tensor_split");
    REQUIRE(ab.policy == "affinity_batch_virtual_tensor_split");
    REQUIRE(ab.tile_virtual_worker.at("t0") ==
            ab.tile_virtual_worker.at("t1"));
    REQUIRE(ab.tile_virtual_worker.at("t2") ==
            ab.tile_virtual_worker.at("t3"));
    if (ab.num_workers >= 2)
    {
        REQUIRE(ab.tile_virtual_worker.at("t0") !=
                ab.tile_virtual_worker.at("t2"));
    }
}
