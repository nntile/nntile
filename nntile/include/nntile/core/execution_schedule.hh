/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/execution_schedule.hh
 * Static task schedule (worker assignment per tile op). ``execution.json`` is
 * produced by an explicit generator (e.g. round-robin) and may be loaded back
 * to configure ``Runtime::execute()``. No data home nodes or MPI ownership.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nntile/core/execution_worker.hh>
#include <nntile/tile/graph_decl.hh>

namespace nntile
{

class Runtime;
class TileGraph;

//! One scheduled tile-graph op after compile-time DCE.
struct ScheduledOpEntry
{
    size_t execution_index = 0;
    std::string op_name;
    std::string task_label;
    int worker = 0;
    std::vector<std::string> writable_tiles;
    std::vector<std::string> read_tiles;
};

//! Must match compiled ``execution_order`` when loading ``execution.json``.
struct ExecutionScheduleFingerprint
{
    size_t op_count = 0;
    std::vector<std::string> op_names;
};

//! Virtual tile split + per-op worker assignment.
struct ExecutionSchedule
{
    std::string policy;
    int num_workers = 1;
    bool use_cuda_workers = false;
    ExecutionScheduleFingerprint fingerprint;
    std::map<std::string, int> tile_virtual_worker;
    std::vector<ScheduledOpEntry> ops;

    int worker_for_op(size_t execution_index) const;
};

ExecutionScheduleFingerprint make_execution_schedule_fingerprint(
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order);

void validate_execution_schedule_fingerprint(
    ExecutionScheduleFingerprint const &fp,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order);

//! Explicit round-robin generator (virtual tensor split + output-owner ops).
ExecutionSchedule generate_round_robin_execution_schedule(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order);

//! Batch-axis affinity: tiles sharing a batch slice use the same worker.
ExecutionSchedule generate_affinity_batch_execution_schedule(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order);

//! Write ``execution.json`` from a schedule (same format as load).
void write_execution_schedule_json(
    ExecutionSchedule const &schedule,
    std::string const &path);

//! Read ``execution.json`` for use with ``Runtime::set_execution_schedule``.
ExecutionSchedule load_execution_schedule_json(std::string const &path);

std::string execution_schedule_to_json(ExecutionSchedule const &schedule);

//! Generate round-robin schedule and write ``execution.json``.
void generate_round_robin_execution_json(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order,
    std::string const &path);

} // namespace nntile
