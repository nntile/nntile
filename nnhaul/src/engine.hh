/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file src/engine.hh
 * RuntimeState over the NodeInventory: the nine-step init's products
 * — the inventory, per-device event pools, the lanes and channels,
 * the host pool, the per-node workspaces and library handles — the
 * registration tables (one executable per task instance and node,
 * scalar-argument blobs as registration-time facts), the pending task
 * chain, the pipeline's seed (the predecessor's checked exit fold), and
 * the engine's acts: close-and-plan, the host acquires
 * as host-node tasks, the dead-handle retirement, and
 * the simulate exposure. No worker thread beyond the host node's lane,
 * no watchdog, no prefetch, no sim world — the previous engine's
 * reactive machinery is deleted with it.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "cuda_backend.hh"
#include "host_pool.hh"
#include "inventory.hh"
#include "lru_planner.hh"
#include "nnhaul/nnhaul.hh"
#include "nnhaul/test_hooks.hh"
#include "plan_executor.hh"
#include "plan_trace.hh"
#include "task.hh"
#include "timing.hh"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nnhaul
{

namespace planner
{
class Planner;
} // namespace planner

// ---- env helpers (parsed once or per-init as noted; the public/env
//      test surface of the previous engine, carried over) ----

inline bool env_set(char const *name)
{
    return std::getenv(name) != nullptr;
}

// Parse a whole decimal integer; false on empty, trailing garbage, or
// out-of-range values (strtol clamps to LONG_MAX and sets ERANGE).
inline bool env_parse_long(char const *e, long &out)
{
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE)
    {
        return false;
    }
    out = parsed;
    return true;
}

// NNHAUL_BOUNDARY_BREAKDOWN, cached once per process through the
// shared value-parsed flag helper (inventory.hh's env_flag_enabled —
// `=0` disables); the boundary brackets read this, never a raw getenv
// per close. Diagnosis only, never control flow.
inline bool boundary_breakdown_on()
{
    static bool const on = env_flag_enabled("NNHAUL_BOUNDARY_BREAKDOWN");
    return on;
}

// NNHAUL_LIMIT_CUDA_MEM as bytes, 0 when unset; garbage is a loud init
// error. Public (the env-robustness test's surface); cuda_backend.cc
// carries its own internal copy for the arena chain.
std::size_t limit_cuda_mem_bytes();

// The cuBLAS workspace size (NNHAUL_CUBLAS_WORKSPACE_MB, cached once
// per process; a parse failure or an out-of-range value clamps to the
// 4-GiB ceiling instead of wrapping the MiB shift).
std::size_t cublas_workspace_bytes();

// THE CAP/WORKSPACE COMPOSITION: the cap (NNHAUL_LIMIT_CUDA_MEM
// or the init argument clamped to 90% free) IS the arena, whole blocks
// only — and the fixed workspaces (gather/scatter,
// cuBLAS, copy descriptors) sit BESIDE the pool, uncounted by the cap
// . The intent is therefore explicit: the cap does
// NOT cover the engine's whole device footprint, and the engine does
// NOT subtract the workspaces from the arena — subtracting would
// change the established composition. The consequence is
// real and is the DEPLOYMENT's to steer: an unled default cap on a
// small card (one giant arena) plus the large workspaces can leave no
// room for the CODELETS' own device scratch (cudaMallocAsync pools live
// outside both), and the lever is NNHAUL_LIMIT_CUDA_MEM. Documented here
// as the knob table's honest
// composition note; no engine-side change.

// NNHAUL_ALLOC_TRACE — the COUNTERS ECHO knob (the old harness's
// observability lineage; this engine neither retires nor
// owns it, so the engine names its one meaning here): when set,
// engine_shutdown prints ONE stderr line after the barriers/joins and
// before resource release, echoing the same counters test_hooks
// read (no new state, no timing effect beyond the one fprintf), and
// every batch close prints one `[nnhaul-plan]` line with the walk's
// own counts (entry/dead-entry/exit residency, evictions, rescues,
// refetches, dying frees) — the churn meters read beside the op
// counters. Counters format: `[nnhaul-counters] kernel_launches=N
// plan_walks=N ops_planned=N ops_issued=N reconcile_ok=N copy_ops=N
// tile_ops=N link_ops=N unlink_ops=N cross_node_copies=N evictions=N
// rescues=N refetches=N dying_frees=N` — stable key=value pairs,
// greppable. (The act itself is declared beside the engine's acts,
// after RuntimeState.)

// The retired knobs of the previous engine: a set knob warns ONCE per
// process and is ignored, so an old environment cannot silently change
// behavior — it says so (NNHAUL_CHUNK_SIZES / NNHAUL_CHUNK_POLICY /
// NNHAUL_CHUNK_MIN_COUNT / NNHAUL_PLAN_CHECK /
// NNHAUL_BATCH_FENCE / NNHAUL_OFFLOAD_POLICY / NNHAUL_LAUNCH_AHEAD).
// (NNHAUL_CHUNK_TILES is not listed: it is NNTile's chunked-TILE
// allocation gate, honored there — see engine.cc.)
void warn_retired_knobs();

inline std::uint64_t now_ns()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// ---- the in-codelet TLS  ----

// The RUNNING node's facts, set at launch dispatch by the registered
// body (the CUDA launch fn on the engine thread, the host resolver on
// the host worker thread) and valid only inside the codelet window:
// the previous single-device TLS (engine.hh:150-162 of the old engine)
// generalized to per-node fields. The host node's entry carries no
// stream and no library handles — those accessors throw inside a CPU
// body; the node id and the host flag are always valid.
struct Tls
{
    bool in_kernel = false;
    bool host_node = false;
    NodeId node = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudnnHandle_t cudnn = nullptr;
    void *workspace = nullptr;
    std::size_t workspace_bytes = 0;
    void *copy_desc = nullptr;
    std::size_t copy_desc_bytes = 0;
    int device = 0; // the running node's CUDA ordinal (0 for the host)
};
extern thread_local Tls tls;

// ---- the registration records  ----

// One operand buffer's registration fact: the tile-side SHAPE the
// Buffer view needs (chunk count, payload bytes) plus the task's
// ACCESS over the buffer — the declared access is a fact of the task
// symbol exactly as the shape is (the checker's correspondence keys
// "the task's access for its tile" per (task, tile), so two instances
// of one codelet declaring DIFFERENT accesses on
// one tile are two symbols, never one). The TILE_OP's operand order —
// per task buffer, chunk indexes ascending — maps the engine's
// operand-address array onto each buffer's chunk span; the symbol's
// capture carries the spans.
struct BufShape
{
    std::size_t n = 0;      // the tile's chunk count
    std::size_t nbytes = 0; // the tile's payload bytes
    Access access = Access::Read; // the task's declared access

    bool operator<(BufShape const &other) const
    {
        if (n != other.n)
        {
            return n < other.n;
        }
        if (nbytes != other.nbytes)
        {
            return nbytes < other.nbytes;
        }
        return static_cast<int>(access) < static_cast<int>(other.access);
    }
    bool operator==(BufShape const &other) const
    {
        return n == other.n && nbytes == other.nbytes &&
               access == other.access;
    }
};

// ONE TASK SYMBOL (this is why symbols are multiplied rather than
// kept per op): the triple (codelet, cl_args blob,
// operand shapes WITH their accesses) IS the symbol. ONE registration
// per (symbol, node), memoized at first sight, never re-registered;
// every instance naming the same triple maps to the same TaskId and
// launches through the same registered body. The cl_args blob, the
// shapes, and the accesses are registration-time facts OF THE SYMBOL
// — which is exactly why they key it: one access per (symbol,
// tile) holds BY CONSTRUCTION, the checker's per-(task, tile)
// presumption satisfied rather than violated. Symbols first seen
// mid-run register at first sight because the workload's inserts ARE
// the registration surface: no instance can exist before its first
// submit.
struct SymbolKey
{
    std::uint32_t codelet = 0;
    std::vector<unsigned char> cl;
    std::vector<BufShape> shapes;

    bool operator==(SymbolKey const &other) const; // engine.cc
};

//! The symbol memoization's O(1) hash:
//! FNV-1a over the codelet id, the cl bytes, and the shapes' triple
//! (n, nbytes, access) — same fields the old std::map comparator
//! walked, hashed once instead of compared down a tree.
struct SymbolKeyHash
{
    std::size_t operator()(SymbolKey const &k) const;
};

//! The (symbol, node) pair's O(1) hash — symbol_registered's key.
struct SymbolNodeHash
{
    std::size_t operator()(std::pair<TaskId, NodeId> const &p) const;
};

// One host-node registration: the CPU body (or the engine's own host
// act — the acquire gather/publish), its cl_args copy,
// and the operand shapes — IMMUTABLE after insertion, served by the
// host worker through the registration mutex while the engine
// registers the next batch's symbols mid-flight (the pipelined
// discipline: the worker may still be serving an earlier batch).
struct HostTaskReg
{
    Codelet *codelet = nullptr;
    std::vector<unsigned char> cl;
    std::vector<BufShape> shapes;
    std::function<void(void *const *)> engine_fn; // null for user tasks
};

// One device-node registration is the CudaLaunchFn itself (memoized
// per (symbol, node), capturing the symbol's facts plus the node's
// TLS facts).

// ---- the per-device fixed workspaces and handles  ----

struct DeviceFacts
{
    int ordinal = 0;
    void *workspace = nullptr;      // gather/scatter, NNHAUL_WORKSPACE_MB
    std::size_t workspace_bytes = 0;
    void *copy_desc = nullptr;      // the 1-MiB copy-descriptor scratch
    std::size_t copy_desc_bytes = 0;
    void *cublas_workspace = nullptr;
    std::size_t cublas_workspace_bytes = 0;
    cublasHandle_t cublas = nullptr;
    cudnnHandle_t cudnn = nullptr;
    std::uint64_t alloc_count = 0; // driver allocations (init acts)
};

// ---- one pinned host acquire run  ----

struct AcquireRun
{
    ChunkId base = 0;   // the contiguous host run's first block
    std::size_t n = 0;  // the tile's chunk count
    bool publish = false; // Write/ReadWrite: the unpin publishes
};

// ---- the runtime ----

struct RuntimeState
{
    // One engine lock: everything but the submitter-side chain runs
    // under it, on the host thread (the host lane's worker serves
    // launches asynchronously but touches only its FIFO and the
    // registration records the engine filled before the enqueue).
    std::mutex mu;
    std::mutex submit_mu; // the pending chain and next_seq

    bool error_set = false;
    //! Process-unique runtime generation; stale Acquired objects carry an
    //! older value and their release is ignored.
    std::uint64_t generation = 0;
    std::string error;

    void set_error(std::string msg)
    {
        if (!error_set)
        {
            error = std::move(msg);
            error_set = true;
        }
    }

    // ---- the runtime's owned resources. TEARDOWN (engine_shutdown)
    //      releases them EXPLICITLY in its own order — the barriers first,
    //      then the host worker's stop, the event pools, the
    //      streams (channels, then the execution lanes with their
    //      arenas), the library handles and fixed workspaces, the host
    //      pool; contexts stay the process exit's act — so this
    //      declaration order carries NO teardown meaning. It serves
    //      INIT's exception unwinding only: a partially built runtime
    //      destroys whatever exists, bridge before the host lane it
    //      parks on, lanes before the pools their streams bind.

    std::vector<int> ordinals;             // step 1: the selected ordinals
    std::vector<std::uint8_t> peer_bits;   // step 1: the probed matrix
    std::unique_ptr<NodeInventory> inv;    // the run's static inventory
    std::vector<std::unique_ptr<CudaEventPool>> event_pools; // step 7
    std::unique_ptr<HostPool> host_pool;   // step 5
    std::unique_ptr<HostWorkerLane> host_lane; // step 6
    std::unique_ptr<DeferralBridge> bridge;    // step 6, after the host lane
    cudaStream_t host_mint_stream = nullptr;  // the mint record
                                              // stream (device 0, empty
                                              // by construction)
    std::vector<std::unique_ptr<CudaExecLane>> exec_lanes; // step 4 (arenas)
    // Dense in channel-lane id order (the inventory numbers the
    // connected pairs in ascending (a, b) order; the executor walks
    // the same order).
    std::vector<std::unique_ptr<TransferChannel>> channels; // step 6
    std::vector<DeviceFacts> dev; // step 8, one per device node

    // The registration tables (run-scoped; ONE entry per TASK SYMBOL
    // — the many instances of one symbol share it — never per instance:
    // intern_symbol memoizes, and the lane tables resolve identity to
    // executable). The host table's mutex: the
    // host worker serves launches through it while the engine
    // registers the next batch's symbols mid-flight (the entries are
    // immutable shared_ptrs, so the serve never holds the lock).
    //! The registration tables are O(1):
    //! unordered containers with hashed keys (the old std::map/set paid
    //! O(log n) key walks per task-symbol memoization, per (symbol,
    //! node) registration dedup, and per plan simulate-check — with
    //! SymbolKey's vector fields compared element-wise down the tree).
    //! No consumer iterates any of these in key order (the symbol
    //! census collects and sorts by symbol id; the body histogram
    //! collects and sorts busiest-first, name-ascending on ties).
    std::unordered_map<TaskId, std::shared_ptr<HostTaskReg const>>
        host_tasks;
    std::mutex host_tasks_mu;
    std::unordered_map<SymbolKey, TaskId, SymbolKeyHash>
        symbol_of; // the memoization
    TaskId host_act_id[3] = {0, 0, 0}; // the engine's host-act symbols
                                       // (1 = gather, 2 = publish) — the
                                       // gate histogram's family names
    std::unordered_set<std::pair<TaskId, NodeId>, SymbolNodeHash>
        symbol_registered;
    std::uint32_t next_symbol = 1; // under mu: the TaskId mint
    std::unordered_map<TaskId, std::uint32_t> task_codelet; // symbol ->
                                                            // codelet

    // The checker's registry inputs, run-scoped: every
    // live handle's tile and every registered instance's runnable
    // nodes.
    TileTable tiles;
    ExecutabilityTable runnable;
    std::unordered_map<TileId, HandleImpl *> live_tiles;

    // Handles awaiting the retirement sweep (invalidate markers).
    // An intrusive doubly-linked list:
    // the links live in HandleImpl (dead_prev/dead_next), so delist
    // runs through the handle's own pointers — O(1). (The vector this
    // replaces paid a
    // linear scan-and-shift per dead handle — the capped cells' top
    // user-space profile symbol, with many dying handles per
    // training iteration.)
    HandleImpl *dead_pending_head = nullptr;
    HandleImpl *dead_pending_tail = nullptr;
    std::size_t dead_pending_count = 0;

    //! O(1) append at the tail. Membership is the caller's test:
    //! dead_prev != nullptr || dead_pending_head == impl.
    void dead_pending_push(HandleImpl *impl);

    //! O(1) delist via the handle's own links; a no-op when the
    //! handle is not listed.
    void dead_pending_unlink(HandleImpl *impl);

    bool dead_pending_empty() const
    {
        return dead_pending_head == nullptr;
    }

    // Active acquire runs, keyed by the handle (the Acquired pin's
    // facts; the unpin consumes them).
    std::unordered_map<HandleImpl *, AcquireRun> acquired;

    TaskArena arena;
    TaskQueue queue;  // append under submit_mu; take at close-and-plan
    std::uint64_t next_seq = 1; // under submit_mu: TaskId mint

    NodeId default_cuda_node = 0; // the default device setter state
    NodeId default_cpu_node = 0;  // the host node (set at init)

    // The pipeline : the seed IS the
    // predecessor's PREDICTED exit fold — advanced at the ISSUE (the
    // successor seeds and issues while the predecessor's lanes still
    // run; enqueue-before-derive is the ENQUEUE order, not the barriers). `unretired` carries the ISSUED, un-joined
    // batch records — at most two (depth-2): a third at a non-empty
    // close is the loud pipeline-depth abort; the join act is the
    // record's queue-close barriers (executor close) plus the
    // retirement accounting below.
    std::vector<EntryBinding> seed;
    std::uint64_t next_plan_seq = 0;
    std::uint64_t batch_counter = 0;

    // One issued, un-joined batch: everything the JOIN consumes — the
    // close's plan and slot, the timing seam's own derivation, the
    // task chain awaiting its Done views, and the retirement that
    // rode the plan.
    struct IssuedBatch
    {
        std::uint64_t batch = 0; // the 1-based batch number
        std::uint64_t plan_seq = 0;
        std::size_t slot = 0;
        Plan plan;
        PlanDerivation deriv;
        Task *tasks = nullptr;      // the arena chain (views at join)
        std::size_t task_log_from = 0; // this batch's Running views
        std::vector<HandleImpl *> swept;
    };
    std::deque<IssuedBatch> unretired;

    planner::Planner *planner_plugin = nullptr; // nullptr = in-tree LRU
    std::unique_ptr<PlanTrace> tracer;
    std::unique_ptr<PlanExecutor> executor;

    // The duration model's loading surface : entries key
    // by CODELET identity and node (set_duration names a codelet, not
    // an instance); the simulate pass consumes an instance-keyed view
    // built per plan. Read-only once the first plan closes.
    std::unordered_map<std::uint64_t, std::uint64_t> duration_by_codelet;
    std::vector<std::uint64_t> channel_rates; // per LaneId, bytes/ns
    DurationModel sim_model;                  // instance-keyed view
    std::unique_ptr<TimingPass> sim_pass;
    std::uint64_t simulate_makespan_ns = 0;
    bool model_read_only = false;
    // The plans the timing pass actually RAN (the exposure is
    // opportunistic — plans naming unmodeled symbols or channels
    // skip). A later plan whose owed sets name an op of a SKIPPED plan
    // skips too: its start rule would demand a simulated end that was
    // never registered (the pass's owed-end-miss is an explicit
    // simulate's census, never this exposure's).
    std::unordered_set<std::uint64_t> simulated_plans;

    // The engine counters the test hooks read .
    std::atomic<std::uint64_t> kernel_launches_{0};
    std::uint64_t batches_planned = 0;
    std::uint64_t planned_ops = 0;
    std::uint64_t link_ops = 0;
    std::uint64_t unlink_ops = 0;
    std::uint64_t copy_ops = 0;
    std::uint64_t tile_ops = 0;
    // The planner's per-plan stats, aggregated (the walk's own counts:
    // evictions taken, rescues landed, refetches delivered, dying
    // frees at the dying positions) — the churn meters read beside
    // the op counters.
    std::uint64_t evictions = 0;
    std::uint64_t rescues = 0;
    std::uint64_t refetches = 0;
    std::uint64_t prefetches = 0;
    std::uint64_t dying_frees = 0;

    // The breakdown knob's per-codelet body histogram (diagnosis
    // only): codelet id -> name and symbol -> that name's node (both
    // recorded once at registration; the nodes are stable pointers),
    // then name -> (busy ns, launches), filled by the device bodies
    // under the mutex when NNHAUL_ISSUE_BREAKDOWN is set. Off by
    // default; never control flow.
    std::unordered_map<std::uint32_t, std::string> codelet_names;
    std::unordered_map<TaskId, std::string const *> symbol_name;
    std::atomic<bool> body_breakdown_{false};
    std::mutex body_mu;
    std::unordered_map<std::string, std::pair<std::uint64_t, std::uint64_t>>
        body_ns;
    std::vector<CopyRecord> copy_log_;
    std::vector<TaskView> task_log;
};

extern std::unique_ptr<RuntimeState> rt;
extern std::mutex api_mu; // guards init/shutdown entry

// The process-global handle registry: handles constructed before init
// must join the runtime's tile table when init runs, so the registry
// outlives any one runtime (defined in api.cc).
void register_handle(HandleImpl *impl);
void unregister_handle(HandleImpl *impl, bool keep_dead = false);
std::vector<HandleImpl *> handle_registry_snapshot();

// ---- the engine's acts (engine.cc) ----

// The nine init steps, each loud on failure; throws nnhaul::Error.
void engine_init(int ncpu, int ncuda, std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each);

// The final join (every un-retired batch), the trace dump,
// then resource release in reverse init order. Never throws.
void engine_shutdown();

// The NNHAUL_ALLOC_TRACE counters echo (the knob's comment above):
// one stderr line from the same counters the test hooks read, after
// every join and before the executor's witnesses and any resource
// release. Called by engine_shutdown only.
void echo_counters_at_shutdown(RuntimeState const &R);

// The close-and-plan ISSUE act: drain
// the pending chain as one batch — seed, intern the symbols, plan,
// check (the FULL inputs, the chain input ALWAYS the current seed),
// ISSUE (the executor enqueues everything and returns; its barriers
// do NOT run here). The batch joins later — join_predecessor() joins
// the older, wait()/the boundary acts join all. A third un-retired
// batch at a non-empty close is the loud pipeline-depth abort. The
// dead-handle retirement is the planner's DYING-VICTIM FREE (the
// dead tiles ride the request; their bindings unlink bare at their
// dying positions inside the batch's own queue, lru_planner's pinned
// order 6) with the trailing retirement sweep over the predicted exit
// kept as the belt that covers anything the walk could not free.
// Throws nnhaul::Error on every census; the runtime remembers the
// error.
GraphEnd engine_issue_batch(RuntimeState &R);

// The JOIN act : the record's queue-close barriers (the
// executor's close — reconcile, barriers device-first/host-last,
// registry sweep, tracer harvest) plus the engine's retirement
// accounting (counters, copy log, the task views' Done flip, the
// simulate over the batch's own owed sets, the swept handles).
// Returns the joined batch's 1-based number, 0 when none stand.
std::uint64_t engine_join_oldest(RuntimeState &R);

// Join every un-retired batch, oldest first. `force` joins past a
// sticky error (shutdown's teardown discipline).
void engine_join_all(RuntimeState &R, bool force = false);

// The boundary acts' fused path (the old engine's close-and-plan with
// wait_after): join all, issue the pending batch (if any), join all —
// wait(), every host acquire, the Handle destructor, and shutdown all
// need a fully joined world.
void engine_sync_close(RuntimeState &R);

// The symbol interning : the
// (codelet, cl_args, shapes) triple's TaskId, memoized at first
// sight; the first sighting registers the executability row and the
// codelet mapping. ensure_symbol_registered() completes the per-node
// registration — the device lane's table (a bridge-ordered step while
// a plan trails the deferral FIFO, so the waiter's concurrent
// launches never race the insert) or the host resolver's table.
// Both require R.mu.
TaskId engine_intern_symbol(RuntimeState &R, Codelet &codelet,
    void const *cl_data, std::size_t cl_size,
    std::vector<BufShape> const &shapes);
void engine_ensure_symbol_registered(RuntimeState &R, TaskId symbol,
    NodeId node, Codelet &codelet, void const *cl_data,
    std::size_t cl_size, std::vector<BufShape> const &shapes);

// The host acquire, re-realized over host-node TILE_OPs:
// the tile's chunks land on a CONTIGUOUS run of whole-class host-pool
// blocks (the Acquired surface hands out one pointer for the whole
// tile) through a plan of exactly the four kinds — LINKs for the run,
// COPYs from the actual replicas, the host gather TILE_OP reading the
// run (a Read acquire) or the host publish TILE_OP writing it (the
// Write/ReadWrite unpin). Every byte move is a plan COPY; nothing
// moves outside a plan. Read/ReadWrite first sync-close (join + drain
// + join); a Write acquire hands the garbage-but-stable run out
// and the unpin publishes it as the tile's actual version. pins += 1
// on return; the pointer is host_pool->addr(base). Throws nnhaul::Error.
void *engine_acquire(RuntimeState &R, HandleImpl &impl, Access access);

// The unpin: pins -= 1, and for a Write/ReadWrite run the publish plan
// (the host publish TILE_OP marking the run actual and outdating every
// other replica) executes here — the user's bytes are already in the
// stable pool blocks, so the op is the publication, not the write.
// Throws nnhaul::Error on a fault; the pin drops regardless.
void engine_release(RuntimeState &R, HandleImpl &impl,
    std::uint64_t generation);

// The Handle destructor's retirement note: place the marker (the
// caller already did), sync-close, sweep. A destructor is noexcept:
// the caller records any failure on the runtime.
void engine_note_retire(RuntimeState &R);
//! True when a pending task or an un-retired batch still names this handle.
bool engine_handle_referenced(RuntimeState &R, Handle const *handle);
//! Mark a swept handle done, unlink it, and free an engine-owned impl.
void engine_release_swept(RuntimeState &R, HandleImpl *impl);

// The retirement sweep's op builder: the dead handles' still-linked
// chunks as UNLINK ops (a dead tile has no remaining reader; the bytes
// go bare). `bindings` is the world the unlinks ride on — the
// planner's predicted EXIT when they trail a batch's ops (the batch
// may have moved a dead tile's replicas), the seed for a retire-only
// plan. Fills `ops` and the swept set.
void engine_retire_ops(RuntimeState &R,
    std::vector<EntryBinding> const &bindings, std::vector<PlanOp> &ops,
    std::vector<HandleImpl *> &swept);

// The batch's symbol pass: intern every pending task's (codelet,
// cl_args, shapes) symbol — memoized — and ensure the per-node
// registration. Fills `symbols` (submit order, instance -> symbol).
// Requires R.mu.
void engine_register_batch(RuntimeState &R, Task *first,
    std::vector<TaskId> &symbols);

// The device launch body factory (the CudaLaunchFn the engine
// registers per (symbol, node): TLS facts, the Buffer views from the
// symbol's shapes, the launch counter, the codelet's CUDA body).
// `body_name` is the breakdown histogram's stable codelet-name node
// (null when unknown — the histogram bucket "?").
CudaLaunchFn engine_device_body(RuntimeState &R, NodeId node,
    Codelet &codelet, void const *cl_data, std::size_t cl_size,
    std::vector<BufShape> shapes,
    std::string const *body_name = nullptr);

// The host lane's resolver (set once at init): the FIFO's Launch items
// call it with the symbol id and the operand addresses; the symbol's
// immutable registration record is served under the registration
// mutex while the engine may be interning the next batch's symbols.
LaunchResolver engine_host_resolver(RuntimeState &R);

// The simulate exposure : at the JOIN, run the
// timing pass over the batch's own folded owed sets when the model
// carries entries for everything the plan names; keep the run makespan
// for the hooks. A miss is not a fault here — the pass simply does not
// advance (the duration-model-miss census belongs to an explicit
// simulate, which the engine does not run unprompted).
void engine_simulate_plan(RuntimeState &R, Plan const &plan,
    PlanDerivation const &deriv);

// The post-join accounting: counters, the copy log. Requires mu.
void engine_note_batch(RuntimeState &R, Plan const &plan,
    PlanExecutor::PlanIssue const &issue);

} // namespace nnhaul
