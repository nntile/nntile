/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file include/nnhaul/test_hooks.hh
 * Diagnostics hooks over the node inventory: per-node pool stats,
 * per-lane busy counters, the copy log, task views, the engine
 * counters with the planned == issued witness and the last
 * lane-clock simulation's run makespan, and the trace knobs — the engine's
 * observable surfaces.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "nnhaul/nnhaul.hh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nnhaul
{

/*! One memory node's pool facts: the current carve the
 * node holds and the binding table's free count over it. Pools are
 * whole chunk_class_bytes() blocks, grown at planning time; the free counts come
 * from the binding table, never the pool.
 * */
struct NodePoolStats
{
    bool host = false;             //!< True for the host node's entry.
    std::size_t pool_bytes = 0;    //!< The carve, grown at planning time.
    std::uint64_t pool_blocks = 0; //!< pool_bytes / the chunk class.
    std::uint64_t free_blocks = 0; //!< The binding table's free count.
    std::size_t free_bytes = 0;    //!< free_blocks * the chunk class.
    std::uint64_t alloc_count = 0; //!< Driver allocations (planning-time acts).
    std::uint64_t free_count = 0;  //!< Driver frees so far.
    /*! Fixed bytes BESIDE the pool: the node's workspace,
     * cuBLAS workspace, and copy-descriptor scratch — never chunks,
     * never counted toward the carve. */
    std::size_t workspace_bytes = 0;
};

/*! Counters shared with the runtime tests, over the node inventory: memory_nodes = k + 1 (devices 0..k-1, host k), one
 * NodePoolStats per node (the host's cpu_cap_bytes real now), and
 * the per-lane busy counters dense over the inventory's lane set
 * (one execution lane per node plus one transfer lane per channel).
 * */
struct Stats
{
    int memory_nodes = 0; //!< k + 1; 0 before init.
    std::size_t cpu_cap_bytes = 0; //!< The host pool bound init chose.
    std::vector<NodePoolStats> node; //!< One entry per node, host last.
    /*! Busy wall time per lane, dense by lane id over the run's lane
     * set (execution and transfer lanes alike). */
    std::vector<std::uint64_t> lane_busy_ns;
    std::uint64_t cross_node_copies = 0; //!< CopyRecords logged so far.
};

/*! The plan engine's counter snapshot (zeros before init): the four
 * op kinds with the planned == issued exactness facts of the batches
 * run so far, and the last lane-clock simulation's run makespan
 * (the contract — the run's predicted end-to-end, the maximum over
 * ALL lane clocks, untouched lanes included).
 * */
struct ManagerStats
{
    std::uint64_t link_ops = 0;   //!< Planned LINK ops issued.
    std::uint64_t unlink_ops = 0; //!< Planned UNLINK ops issued.
    std::uint64_t copy_ops = 0;   //!< Planned COPY ops issued.
    std::uint64_t tile_ops = 0;   //!< Planned TILE_OP ops issued.
    std::uint64_t plan_walks = 0; //!< Batches planned (close-and-plan).
    /*! Ops planned across all batches — with plan_ops_issued the
     * exactness witness the trace's metadata block also carries. */
    std::uint64_t plan_ops_planned = 0;
    std::uint64_t plan_ops_issued = 0; //!< Ops the executor issued.
    bool plan_reconcile_ok = true; //!< planned == issued, every batch.
    /*! The last lane-clock simulation's RUN makespan in ns (0 before
     * any plan): local makespans measure plans, this measures the
     * run. */
    std::uint64_t simulate_makespan_ns = 0;
};

/*! Lifecycle phase of a submitted task. */
enum class TaskPhase
{
    Submitted, //!< Queued, not yet in a closed plan.
    Running,   //!< Its plan issued; the op not yet joined.
    Done,      //!< Joined at the queue-close barrier.
    Failed     //!< The runtime errored on this task.
};

/*! One task as the tests see it (snapshot of the pending chain and
 * history): the resolved node id recorded at submit (0..k), the
 * diagnostic kind, and the footprint key. */
struct TaskView
{
    std::uint64_t submit_seq = 0;  //!< Submission sequence number.
    WorkerId worker = worker_unspecified; //!< The resolved node id.
    Kind kind = Kind::Cuda;        //!< The node's class (diagnostic).
    std::uint64_t footprint = 0;   //!< The diagnostic history key.
    TaskPhase phase = TaskPhase::Submitted; //!< Current phase.
    std::string codelet;           //!< Codelet diagnostic name.
};

/*! One logged transfer (what cross_node_copies counts): a planned
 * COPY as issued, the occupied span of one logical chunk over one
 * channel. */
struct CopyRecord
{
    /*! The run-scoped op reference: (plan_seq << 32) | queue
     * position. */
    std::uint64_t op_ref = 0;
    std::string handle;           //!< Handle diagnostic name.
    int src_node = -1;            //!< Node id over 0..k.
    int dst_node = -1;            //!< Node id over 0..k.
    /*! Bytes of THIS copy: the occupied span of the copied logical
     * chunk — min(chunk_class_bytes(), nbytes - index*chunk_class_bytes()),
     * the contract's byte rule. A full chunk moves the whole class; a tail
     * chunk moves exactly its tile payload's remainder. */
    std::size_t nbytes = 0;
};

/*! Pool and lane counter snapshot (zeros before init). */
Stats stats();

/*! The plan engine's counters (zeros before init). */
ManagerStats manager_stats();

/*! Completed task history plus the still-queued tail. */
std::vector<TaskView> tasks();

/*! True once the handle's invalidate has been swept: its bindings
 * are unlinked and the data is gone. */
bool handle_swept(Handle const &handle);

/*! Every logged transfer so far. */
std::vector<CopyRecord> copy_log();

/*! The node id inside a codelet body; worker_unspecified outside. */
WorkerId current_worker();

/*! The compute kind inside a codelet body (the running node's
 * class); Kind::Cpu outside. */
Kind current_kind();

/*! Kernel launches since init (the per-node launch counter's sum). */
std::uint64_t kernel_launches();

/*! True while tracing is armed: a runtime exists and NNHAUL_TRACE
 * enabled it at init. False before init, after shutdown, and for the
 * whole knob-off state (unset, `0`, or empty). */
bool trace_enabled();

/*! Write what the trace accumulated so far to `path` as Chrome Trace
 * Event JSON — the same format as the NNHAUL_TRACE shutdown dump —
 * and keep accumulating. False when tracing is off, no runtime
 * exists, or the write failed (the failure also reports to stderr). */
bool trace_dump(std::string const &path);

} // namespace nnhaul
