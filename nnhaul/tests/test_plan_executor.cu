/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * execute-mode issue loop on REAL lanes over one GPU and the host node.
 *
 * @file tests/test_plan_executor.cu
 * The issue loop's suite (ctest target plan_executor): the classic
 * 1-GPU+host inventory with real lanes built from src/cuda_backend.cc +
 * src/host_worker.cc, canary launch fns registered per node (a byte-fill
 * and a byte-compare on each side), and a full plan (LINKs, a
 * host-staged COPY pair both directions, TILE_OPs on BOTH exec lanes,
 * UNLINKs) with byte-level verification through the DERIVED syncs
 * (the host reader sees the device writer's bytes only through the
 * COPY and its wait; the device reader likewise through the host
 * producer's gate); planned == issued (the issue log equals the queue
 * op for op, the issued waits equal the derived owed set, the mint
 * counts acquired once and released at the sweep by displacement);
 * boundary-mismatch on a one-flag-flip entry aborting BEFORE any op
 * issues (no stamp, no fire, no issue); issue-abort on an unregistered
 * TaskId with the census naming op + OpRef and the later canary never
 * enqueued; registry-miss on a hand-broken wait (a registry entry
 * wiped between derive and issue through the documented test seam);
 * the late-record path (plan 2 owing plan 1's op — the mint lands in
 * plan 1's lane at plan 2's derivation, the bytes still ordered); the
 * plan-seq continuity demand; and, with the tracer armed, the harvest
 * yielding a measured record for every lane-carrying op with
 * end >= start (no gating); the byte-exact copy check (the byte rule at the
 * dispatch): a TAIL chunk's COPY transfers exactly the tile
 * payload's occupied span — the bytes prove it (the block's wasted
 * tail keeps the stale plant), the recording proxies report the byte
 * count handed each channel (full chunk the whole class, tail chunk
 * the remainder), and the tracer's byte stamp names the span. The
 * deferral bridge's own three: the
 * mixed-plan repro (two host-gated h2d copies around a d2h'd device
 * write consumed on the host — the exact shape that deadlocked when
 * the host consumer's wait bypassed the single issue order), a refusal
 * INSIDE a deferred step
 * (the unregistered task lands after the first gate — the census at
 * the barrier names the step's exact op via the bridge tag,
 * dependents never run), the fault-aware release (a host-lane
 * codelet fault behind a parked gate — the gate surfaces, the census
 * delivers at the barrier, the deferred canary never runs), and the
 * device-fault drain-park (an illegal-address kernel whose fault
 * kills the notify a parked gate's chain waits on — the sync-first
 * close delivers the census naming the faulting op, no hang; a
 * forked, alarm-guarded child so a regression fails the check). GPU:
 * the
 * suite initializes CUDA — run under the suite's GPU lock.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/cuda_backend.hh"
#include "../src/host_pool.hh"
#include "../src/plan_executor.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today — the standalone CHECK macro stands here,
// same shape as test_plan_trace.cc's and test_timing.cc's.
#include "nnhaul/defs.h"

#include <cuda_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace nnhaul;

int g_checks_failed = 0;

void check_impl(bool ok, char const *expr, char const *file, int line)
{
    if (!ok)
    {
        std::cerr << "CHECK failed: " << expr << " (" << file << ":" << line
                  << ")\n";
        g_checks_failed += 1;
    }
}

#define CHECK(cond)                                                           \
    check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

// A loud-census probe: runs f, true iff it threw ExecutorCensusError,
// and when `needle` is given also demands the census text carry it.
template <typename F>
bool throws_census(F &&f, char const *needle = nullptr)
{
    try
    {
        f();
    }
    catch (ExecutorCensusError const &e)
    {
        if (needle != nullptr &&
            e.census.fault.find(needle) == std::string::npos)
        {
            std::cerr << "CHECK failed: the census lacks '" << needle
                      << "': " << e.census.fault << "\n";
            g_checks_failed += 1;
        }
        return true;
    }
    catch (std::exception const &e)
    {
        std::cerr << "CHECK failed: a foreign exception: " << e.what()
                  << "\n";
        g_checks_failed += 1;
        return true;
    }
    return false;
}

// ---- the canary kernels (the launch fn IS allowed to enqueue
//      anything on its lane's stream) ----

// Writes ok = 0 when any byte of the chunk differs from `expected`
// (the flag is planted at 1 before the plan — a read that observes the
// producer's bytes through the derived syncs leaves it 1).
__global__ void k_check_eq(
    unsigned char const *p, unsigned char expected, std::size_t n, int *ok)
{
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        if (p[i] != expected)
        {
            *ok = 0;
        }
    }
}

// The device-fault canary: spins ~a few hundred ms, then commits an
// illegal access — every enqueue of the plan (and the close join's
// round one) happens while the kernel still RUNS, healthy; the sync
// then delivers the fault deterministically instead of racing the
// sticky error against the issue loop's enqueue calls.
__global__ void k_fault_delayed(int *p)
{
    long long const until = clock64() + 600000000LL;
    while (clock64() < until)
    {
    }
    *p = 1; // p == nullptr: the illegal-address fault
}

// ---- the classic inventory and its real lanes  ----

// NodeIds: device 0 = 0, host = 1 . Lanes: exec(0) = 0,
// exec(host) = 1, chan(0,1) = 2 (gpu -> host), chan(1,0) = 3
// (host -> gpu) — four lanes, the classic reduction.
constexpr NodeId kGpu = 0;
constexpr NodeId kHost = 1;
constexpr LaneId kLaneExecGpu = 0; // exec(0): the gpu's lane id
constexpr std::size_t kChunksPerNode = 4;
constexpr std::size_t kProvision = 64;

// The span-check scenario's tiles: one TAIL tile (3/4 of the class — one
// chunk, its payload short of the block) and one FULL tile (exactly
// one class), registered per scenario into the fixture's own table.
constexpr TileId kTailTile = 40;
constexpr TileId kFullTile = 41;
// The COPY-run seam's two-chunk tile: both chunks full-class, so two
// consecutive same-lane COPYs of its chunks are address-contiguous on
// BOTH endpoints — the run-merge shape.
constexpr TileId kTwoTile = 42;

// The registered canary tasks. The GPU side fills a whole chunk with
// one byte or byte-compares it into a pre-planted device flag; the
// host side fills pinned bytes or scans them into a test cell. Scalar
// facts (the byte, the flag cell) are registration-time facts of the
// task symbol, never per-op plan fields.
constexpr TaskId kGpuFillAA = 30;
constexpr TaskId kGpuFillBB = 31;
constexpr TaskId kGpuFill5A = 32;
constexpr TaskId kGpuCheckCC = 40;
constexpr TaskId kHostFillCC = 10;
constexpr TaskId kHostFault = 12; // throws: the host-lane fault canary
constexpr TaskId kHostCheckAA = 20;
constexpr TaskId kHostCheck5A = 21;
constexpr TaskId kHostCheck33 = 22;
constexpr TaskId kHostFill11 = 13;
constexpr TaskId kHostFill22 = 14;
constexpr TaskId kGpuFill33 = 34;
constexpr TaskId kGpuFaultBad = 35; // the delayed illegal-address kernel

// The span check's recording proxy over a real channel: RECORDS every
// copy's submitted byte count beside performing the real transfer
// through the wrapped channel — every ordering rule delegates
// verbatim, so the executor's mint/wait/stamp/barrier machinery sees
// the real lane and only the copy's byte handoff is witnessed.
struct RecordingCopy
{
    void const *src = nullptr;
    void *dst = nullptr;
    std::size_t nbytes = 0;
};

class RecordingChannel final : public TransferChannel
{
  public:
    explicit RecordingChannel(CudaHostChannel &real) : real_(real)
    {
    }

    void copy(void const *src, void *dst, std::size_t nbytes) override
    {
        copies.push_back(RecordingCopy{src, dst, nbytes});
        real_.copy(src, dst, nbytes); // the real span-exact transfer
    }

    // The run-merge seam's witness: one entry per MERGED driver
    // call. `default_slice` routes the run through the BASE class's
    // realization — class-sized copy() slices — instead of the CUDA
    // channel's one-call form, so the default path's equivalence is
    // asserted on the very same recording (the slices land in
    // `copies`, the bytes still fly through the real channel).
    void copy_run(void const *src, void *dst, std::size_t nbytes) override
    {
        runs.push_back(RecordingCopy{src, dst, nbytes});
        if (default_slice)
        {
            TransferChannel::copy_run(src, dst, nbytes);
        }
        else
        {
            real_.copy_run(src, dst, nbytes);
        }
    }

    void mint(EventHandle e) override { real_.mint(e); }
    void wait(EventHandle e) override { real_.wait(e); }
    void notify(NotifyHook fn, void *arg) override
    {
        real_.notify(fn, arg);
    }
    EventHandle stamp_acquire() override { return real_.stamp_acquire(); }
    void stamp_release(EventHandle h) override
    {
        real_.stamp_release(h);
    }
    void stamp(EventHandle h) override { real_.stamp(h); }
    StampFact stamp_read(EventHandle before, EventHandle after) override
    {
        return real_.stamp_read(before, after);
    }
    void barrier() override { real_.barrier(); }

    std::vector<RecordingCopy> copies;
    std::vector<RecordingCopy> runs; // one entry per copy_run call
    bool default_slice = false;      // run through the base slicer

  private:
    CudaHostChannel &real_;
};

struct Fx
{
    NodeInventory inv;
    CudaEventPool pool;
    CudaExecLane gpu;
    HostWorkerLane host;
    DeferralBridge bridge; // after the host lane (it parks on the
                           // lane's flags), before the CUDA lanes
    HostPool hpool;
    CudaHostChannel d2h; // chan(0,1): gpu -> host
    CudaHostChannel h2d; // chan(1,0): host -> gpu
    // The tile table: the registered payload bytes the COPY spans
    // derive from (the executor's dispatch and the tracer's
    // registration both read it). The suite's pre-existing
    // plans COPY tiles 0, 1, 2 and 9, every one a full-class tile;
    // the span-check tiles register themselves per scenario.
    TileTable tiles;
    std::unique_ptr<PlanTrace> tracer;
    std::unique_ptr<PlanExecutor> ex;
    std::unique_ptr<RecordingChannel> rec_d2h; // the span-check proxies
    std::unique_ptr<RecordingChannel> rec_h2d; // (recording builds)
    cudaStream_t mint_stream = nullptr; // the twins' record stream

    std::atomic<int> host_check_result{0};
    int *d_check_cc = nullptr; // kGpuCheckCC's flag cell

    static bool scan_eq(void const *p, unsigned char byte)
    {
        auto const *q = static_cast<unsigned char const *>(p);
        for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
        {
            if (q[i] != byte)
            {
                return false;
            }
        }
        return true;
    }

    Fx(bool arm_trace = false, bool recording = false) :
        inv(1, {kChunksPerNode, kChunksPerNode}, {0}),
        pool(0, kProvision),
        gpu(0, 0, kChunksPerNode, pool),
        host(kProvision,
            [this](TaskId t, void *const *addrs) {
                switch (t)
                {
                case kHostFillCC:
                    std::memset(addrs[0], 0xCC, chunk_class_bytes());
                    break;
                case kHostFill11:
                    std::memset(addrs[0], 0x11, chunk_class_bytes());
                    break;
                case kHostFill22:
                    std::memset(addrs[0], 0x22, chunk_class_bytes());
                    break;
                case kHostFault:
                    throw std::runtime_error(
                        "host codelet fault (the canary)");
                case kHostCheck33:
                    host_check_result.store(
                        scan_eq(addrs[0], 0x33) ? 1 : 0);
                    break;
                case kHostCheckAA:
                    host_check_result.store(
                        scan_eq(addrs[0], 0xAA) ? 1 : 0);
                    break;
                case kHostCheck5A:
                    host_check_result.store(
                        scan_eq(addrs[0], 0x5A) ? 1 : 0);
                    break;
                default:
                    throw std::runtime_error(
                        "host resolver: unknown task (a loud lane fault)");
                }
            }),
        bridge(host),
        hpool(0, kChunksPerNode), // unlimited, preallocated
        d2h(0, HostChannelDirection::DeviceToHost, pool),
        h2d(0, HostChannelDirection::HostToDevice, pool)
    {
        // The registered bodies: everything they enqueue rides
        // the lane's own stream.
        gpu.register_task(kGpuFillAA,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0xAA, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuFillBB,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0xBB, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuFill5A,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0x5A, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuFill33,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0x33, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuFaultBad,
            [](cudaStream_t s, void *const *) {
                k_fault_delayed<<<1, 1, 0, s>>>(nullptr);
            });

        // The device flag cell for the byte-compare, planted at the
        // int VALUE 1 (cudaMemset would write 0x01010101, not 1).
        if (cudaMalloc(&d_check_cc, sizeof(int)) != cudaSuccess)
        {
            throw std::runtime_error("fixture: the check flag's setup");
        }
        int const one = 1;
        if (cudaMemcpy(d_check_cc, &one, sizeof(int),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("fixture: the check flag's plant");
        }
        gpu.register_task(kGpuCheckCC,
            [this](cudaStream_t s, void *const *a) {
                k_check_eq<<<1, 256, 0, s>>>(
                    static_cast<unsigned char const *>(a[0]), 0xCC,
                    chunk_class_bytes(), d_check_cc);
            });

        // The tile rows : one registration per tile the
        // suite's plans COPY, each a full-class payload — the span the
        // dispatch computes from it is the whole class.
        tiles.add(0, chunk_class_bytes());
        tiles.add(1, chunk_class_bytes());
        tiles.add(2, chunk_class_bytes());
        tiles.add(9, chunk_class_bytes());

        build_executor(arm_trace, recording);
    }

    ~Fx()
    {
        // The lanes drain at every successful close; after an abort the
        // fixture joins them itself before the members unwind (an
        // outstanding host gate is released by the lane's own teardown).
        try
        {
            gpu.barrier();
        }
        catch (...)
        {
        }
        try
        {
            d2h.barrier();
        }
        catch (...)
        {
        }
        try
        {
            h2d.barrier();
        }
        catch (...)
        {
        }
        if (d_check_cc != nullptr)
        {
            cudaFree(d_check_cc);
        }
    }

    Fx(Fx const &) = delete;
    Fx &operator=(Fx const &) = delete;

    // (Re)builds the executor over the lanes; with_tracer arms the
    // stamps ;
    // with_recording the transfer channels sit behind the span check's
    // recording proxies (the real transfers still fly through them).
    void build_executor(bool with_tracer, bool with_recording = false)
    {
        if (with_tracer)
        {
            tracer = std::make_unique<PlanTrace>(inv, tiles);
            tracer->arm("/tmp/nnhaul-plan-executor-trace.json");
        }
        std::vector<ExecLane *> execs{&gpu, &host};
        std::vector<TransferChannel *> chans;
        if (with_recording)
        {
            rec_d2h = std::make_unique<RecordingChannel>(d2h);
            rec_h2d = std::make_unique<RecordingChannel>(h2d);
            chans = {rec_d2h.get(), rec_h2d.get()};
        }
        else
        {
            chans = {&d2h, &h2d};
        }
        std::vector<EventPool *> pools{&pool};
        std::vector<ChunkAddresser> addr{
            [this](ChunkId c) { return gpu.addr(c); },
            [this](ChunkId c) { return hpool.addr(c); },
        };
        // The twins' seam: device events from the
        // fixture's own pool, recorded by the host worker on a
        // dedicated empty stream at the mint's FIFO position; a hook
        // failure releases the recorded-wait parks (the deadlock
        // closure) before it faults the lane.
        if (mint_stream == nullptr &&
            cudaStreamCreateWithFlags(
                &mint_stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            throw std::runtime_error("fixture: the twins' stream");
        }
        ex = std::make_unique<PlanExecutor>(inv, execs, &host, chans,
            pools, std::move(addr),
            std::vector<ChunkContig>{
                [](ChunkId) { return chunk_class_bytes(); },
                [](ChunkId) { return chunk_class_bytes(); }},
            &bridge, &pool,
            [this](EventHandle e) {
                cudaError_t const err =
                    cudaEventRecord(pool.raw(e), mint_stream);
                if (err != cudaSuccess)
                {
                    bridge.release_recorded_waits(
                        "the fixture's mint twin was refused: " +
                        std::string(cudaGetErrorString(err)));
                    throw std::runtime_error(
                        std::string("fixture mint twin: ") +
                        cudaGetErrorString(err));
                }
            },
            tiles, tracer.get());
    }

    int read_check_cc()
    {
        int v = -1;
        if (cudaMemcpy(&v, d_check_cc, sizeof(int),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("fixture: the check flag's read");
        }
        return v;
    }
};

// ---- op factories local to this suite ----

PlanOp link(TileId t, ChunkIndex i, NodeId n, ChunkId c)
{
    return make_link(t, i, n, c);
}

PlanOp unlink(TileId t, ChunkIndex i, NodeId n, ChunkId c)
{
    return make_unlink(t, i, n, c);
}

PlanOp copy(TileId t, ChunkIndex i, ChunkLoc s, ChunkLoc d)
{
    return make_copy(t, i, s, d);
}

PlanOp tile(TaskId task, NodeId n, std::vector<Operand> ops)
{
    return make_tile_op(task, n, std::move(ops));
}

Operand opnd(TileId t, Access a, NodeId n, ChunkId c)
{
    Operand o;
    o.tile = t;
    o.index = 0;
    o.access = a;
    o.node = n;
    o.chunk = c;
    return o;
}

// opnd pins index 0 (every pre-existing scenario's tiles are
// single-chunk); the run-merge scenario's two-chunk tile needs the
// chunk's own index.
Operand opnd_ix(TileId t, ChunkIndex ix, Access a, NodeId n, ChunkId c)
{
    Operand o = opnd(t, a, n, c);
    o.index = ix;
    return o;
}

// ---- 1. the full plan: bytes through the derived syncs, planned ==
//         issued, mint counts, barriers, stamps/fire ----

void scenario_full_plan()
{
    Fx fx(true); // the armed tracer: the stamps/fire scenario rides it

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(0, 0, kGpu, 0),  // 0: T0 lands on the gpu chunk
        link(0, 0, kHost, 1), // 1: T0's host replica (the D2H dst)
        link(1, 0, kHost, 2), // 2: T1 born on host (the H2D src)
        link(1, 0, kGpu, 3),  // 3: T1's gpu replica (the H2D dst)
        tile(kGpuFillAA, kGpu,
            {opnd(0, Access::Write, kGpu, 0)}),                    // 4
        copy(0, 0, {kGpu, 0}, {kHost, 1}),                         // 5
        tile(kHostCheckAA, kHost, {opnd(0, Access::Read, kHost, 1)}), // 6
        tile(kHostFillCC, kHost, {opnd(1, Access::Write, kHost, 2)}), // 7
        copy(1, 0, {kHost, 2}, {kGpu, 3}),                         // 8
        tile(kGpuCheckCC, kGpu, {opnd(1, Access::Read, kGpu, 3)}), // 9
        unlink(0, 0, kHost, 1),                                    // 10
        unlink(1, 0, kGpu, 3),                                     // 11
        unlink(0, 0, kGpu, 0),                                     // 12
        unlink(1, 0, kHost, 2),                                    // 13
    };

    OpRef const r4 = make_op_ref(0, 4);
    OpRef const r5 = make_op_ref(0, 5);
    OpRef const r7 = make_op_ref(0, 7);
    OpRef const r8 = make_op_ref(0, 8);

    fx.ex->execute(p);

    // The bytes — read only after execute() returned, i.e. after the
    // queue-close barriers drained every lane (the barrier proof: the
    // host worker's own write is visible, the device flag readable).
    CHECK(fx.host_check_result.load() == 1); // 0xAA arrived through 5+6
    CHECK(fx.read_check_cc() == 1);          // 0xCC arrived through 7+8+9

    // The derivation's shape, pinned by hand (the one dependency pass):
    // p5 owes p4 (chan waits exec), p6 owes p5 (host waits a device
    // mint — the host-flag path), p8 owes p7 (a DEVICE channel defers
    // on a HOST mint — the deferred-enqueue path), p9 owes p8 (exec
    // waits chan).
    PlanDerivation const &d = fx.ex->last_derivation();
    CHECK(d.mints[4] && d.mints[5] && d.mints[7] && d.mints[8]);
    CHECK(!d.mints[0] && !d.mints[1] && !d.mints[2] && !d.mints[3]);
    CHECK(!d.mints[6] && !d.mints[9] && !d.mints[10] && !d.mints[13]);
    CHECK(d.late_records.empty());
    std::vector<std::vector<OpRef>> const expected_waits = {
        {},          // 0 LINK
        {},          // 1 LINK
        {},          // 2 LINK
        {},          // 3 LINK
        {},          // 4 the gpu fill owes nothing
        {r4},        // 5 the COPY waits the gpu fill's mint
        {r5},        // 6 the host reader waits the COPY's mint (flag)
        {},          // 7 the host fill owes nothing
        {r7},        // 8 the h2d COPY defers on the host fill's mint
        {r8},        // 9 the gpu reader waits the COPY's mint
        {},          // 10 UNLINK
        {},          // 11 UNLINK
        {},          // 12 UNLINK
        {},          // 13 UNLINK
    };
    CHECK(fx.ex->last_issue().issued_waits.size() == 14);
    CHECK(fx.ex->last_issue().issued_waits == expected_waits);

    // planned == issued, both halves: the log op for op (value
    // equality) and the sync half against the derivation itself.
    CHECK(fx.ex->last_issue().issue_log.size() == p.queue.size());
    for (std::size_t i = 0; i < p.queue.size(); ++i)
    {
        CHECK(fx.ex->last_issue().issue_log[i] == p.queue[i]);
    }
    CHECK(fx.ex->last_issue().issued_waits == d.owed_set);
    CHECK(fx.ex->last_issue().reconcile_ok);
    CHECK(fx.ex->plans_executed() == 1);
    CHECK(fx.ex->ops_issued() == 14);

    // Mint counts: four mints acquired once each (4, 5, 7, 8), none
    // displaced at the sweep (every minted op still carries a USER
    // fact — the facts never reset), so the registry
    // holds all four and nothing was released.
    CHECK(fx.ex->mints_acquired_total() == 4);
    CHECK(fx.ex->last_issue().mints_enqueued == 4);
    CHECK(fx.ex->handles_released_total() == 0);
    CHECK(fx.ex->registry().map.size() == 4);
    CHECK(fx.ex->mints_acquired_total() - fx.ex->handles_released_total() ==
        fx.ex->registry().map.size());

    // The applied table: the exit fold — everything unlinked, all
    // pools free again.
    CHECK(fx.ex->applied().snapshot().empty());
    CHECK(fx.ex->applied().free_count(kGpu) == kChunksPerNode);
    CHECK(fx.ex->applied().free_count(kHost) == kChunksPerNode);

    // Stamps and fires (armed tracer): a measured record for every
    // lane-carrying op — the four TILE_OPs and the two COPYs — each
    // end >= start (a fact, gating nothing).
    CHECK(fx.tracer->fired_ops() == 6);
    CHECK(fx.tracer->measured_records().size() == 6);
    for (MeasuredTraceRecord const &r : fx.tracer->measured_records())
    {
        CHECK(r.end_ns >= r.start_ns);
        CHECK(r.lane != kNoLane);
    }
}

// ---- 2. boundary-mismatch: one flag flip, nothing issues ----

void scenario_boundary_mismatch()
{
    Fx fx(true); // armed: the no-op-issued witness

    // The run's initial load through the applied table's own
    // transitions : T7 linked and ACTUAL on the gpu chunk.
    fx.ex->applied().link(kGpu, 0, LogicalChunk{7, 0});
    fx.ex->applied().actualize(kGpu, 0);

    // The plan's entry state: the same binding with the flag FLIPPED
    // — the one hand-broken bit, on one node's segment.
    Plan p;
    p.plan_seq = 0;
    EntryBinding b;
    b.node = kGpu;
    b.chunk = 0;
    b.tile = 7;
    b.index = 0;
    b.actual = false; // the applied table says actual — the flip
    p.entry = {b};
    p.queue = {link(8, 0, kGpu, 1),
        tile(kGpuFillAA, kGpu, {opnd(8, Access::Write, kGpu, 1)}),
        unlink(8, 0, kGpu, 1)};

    CHECK(throws_census([&] { fx.ex->execute(p); }, "boundary-mismatch"));
    // BEFORE any op issues: no stamp, no fire, no issue, no plan.
    CHECK(fx.ex->ops_issued() == 0);
    CHECK(fx.ex->plans_executed() == 0);
    CHECK(fx.tracer->measured_records().empty());
    CHECK(fx.tracer->fired_ops() == 0);
    CHECK(fx.ex->mints_acquired_total() == 0);
}

// ---- 3. issue-abort: an unregistered TaskId; dependents never run ----

void scenario_issue_abort()
{
    Fx fx;

    // The canary chunk pre-zeroed through the lane's own surface, so
    // "never ran" reads as still-zero, not as garbage.
    unsigned char *zero = static_cast<unsigned char *>(fx.gpu.addr(1));
    // A sync cudaMemset's host-block does not order against the
    // lanes' non-blocking streams under device contention (the CI
    // flake of run 41464137985: a deferred plant executed after the
    // work it plants against). Join the legacy stream the memset
    // runs on before the plan's lane work is issued.
    CHECK(cudaMemset(zero, 0, chunk_class_bytes()) == cudaSuccess);
    CHECK(cudaStreamSynchronize(0) == cudaSuccess);

    constexpr TaskId kUnregistered = 77; // no body on any lane
    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(0, 0, kGpu, 0),                    // 0 (issues)
        link(3, 0, kGpu, 1),                    // 1 (issues)
        tile(kUnregistered, kGpu,
            {opnd(3, Access::Write, kGpu, 1)}), // 2: refused at enqueue
        tile(kGpuFillBB, kGpu,
            {opnd(3, Access::Write, kGpu, 1)}), // 3: the never-run canary
        unlink(3, 0, kGpu, 1),                  // 4: never enqueued
        unlink(0, 0, kGpu, 0),                  // 5: never enqueued
    };

    bool threw = false;
    Census caught;
    try
    {
        fx.ex->execute(p);
    }
    catch (ExecutorCensusError const &e)
    {
        threw = true;
        caught = e.census;
    }
    CHECK(threw);
    // The census names the op: the fault, the position, the OpRef, the
    // lane's own detail (the task id).
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    CHECK(caught.fault.find("position 2") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=2)") !=
        std::string::npos);
    CHECK(caught.fault.find("77") != std::string::npos);
    CHECK(caught.op.has_value());
    CHECK(caught.op->kind == PlanOpKind::TileOp);
    // The issue log prefix: the two LINKs issued, nothing after.
    CHECK(caught.issue_prefix.size() == 2);
    CHECK(fx.ex->ops_issued() == 2);
    CHECK(fx.ex->plans_executed() == 0);

    // Dependents never ran: the fixture drains the lanes itself (the
    // executor never closes an aborted plan) and the canary bytes stay
    // zero.
    fx.gpu.barrier();
    std::vector<unsigned char> back(chunk_class_bytes(), 1);
    CHECK(cudaMemcpy(back.data(), zero, chunk_class_bytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    bool any = false;
    for (unsigned char v : back)
    {
        any = any || v != 0;
    }
    CHECK(!any); // the fill of position 3 never enqueued
}

// ---- 4. registry-miss: a hand-broken wait through the test seam ----

void scenario_registry_miss()
{
    Fx fx;

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(0, 0, kGpu, 0), // 0
        link(0, 0, kHost, 1), // 1
        tile(kGpuFillAA, kGpu, {opnd(0, Access::Write, kGpu, 0)}), // 2
        copy(0, 0, {kGpu, 0}, {kHost, 1}),                         // 3
        tile(kHostCheckAA, kHost, {opnd(0, Access::Read, kHost, 1)}), // 4
    };

    // The documented seam: wipe the producer's mint between the
    // derivation and the issue — position 3's wait (a DEVICE consumer)
    // then names an absent entry.
    fx.ex->set_post_derivation_hook(
        [&fx](Plan const &plan, PlanDerivation const &deriv) {
            (void)deriv;
            fx.ex->registry().map.erase(make_op_ref(plan.plan_seq, 2));
        });

    CHECK(throws_census([&] { fx.ex->execute(p); }, "registry-miss"));
    // The census names the consuming op (position 3, the COPY) and the
    // prefix carries the three issued ops before it.
    CHECK(fx.ex->ops_issued() == 3);
}

// ---- 5. the late-record path: plan 2 owes plan 1's op ----

void scenario_late_record()
{
    Fx fx;

    // Plan 1: the gpu writes T0 with no cross-lane consumer in the
    // plan — nothing mints, the facts carry the producer forward.
    Plan p1;
    p1.plan_seq = 0;
    p1.queue = {
        link(0, 0, kGpu, 0),
        link(0, 0, kHost, 1),
        tile(kGpuFill5A, kGpu, {opnd(0, Access::Write, kGpu, 0)}),
    };
    fx.ex->execute(p1);
    CHECK(fx.ex->mints_acquired_total() == 0); // nothing owed across
    CHECK(fx.ex->registry().map.empty());

    // Plan 2's entry IS the predecessor's applied exit fold (the
    // successor seeds from the exit state).
    Plan p2;
    p2.plan_seq = 1;
    p2.entry = fx.ex->applied().snapshot();
    CHECK(p2.entry.size() == 2);
    p2.queue = {
        copy(0, 0, {kGpu, 0}, {kHost, 1}), // owes plan 1's fill
        tile(kHostCheck5A, kHost, {opnd(0, Access::Read, kHost, 1)}),
    };
    fx.ex->execute(p2);

    // The late record: one cross-plan mint, enqueued into the OWED
    // op's lane (the gpu exec lane) at THIS derivation pass — and the
    // host consumer's wait orders through it (the bytes prove it).
    PlanDerivation const &d = fx.ex->last_derivation();
    CHECK(d.late_records.size() == 1);
    CHECK(d.late_records[0].op == make_op_ref(0, 2));
    CHECK(d.late_records[0].owner == kLaneExecGpu);
    // Two handles this plan: the late record AND the COPY's own mint
    // (the host consumer of plan 2 owes the COPY — minted within the
    // plan).
    CHECK(fx.ex->last_issue().mints_enqueued == 2);
    CHECK(fx.ex->mints_acquired_total() == 2);
    CHECK(fx.host_check_result.load() == 1); // 0x5A through the wait
    CHECK(fx.ex->last_issue().issue_log.size() == 2);
    CHECK(fx.ex->plans_executed() == 2);

    // The continuity demand: a plan_seq gap is loud, never serviced.
    Plan p3;
    p3.plan_seq = 3;
    p3.entry = fx.ex->applied().snapshot();
    CHECK(throws_census(
        [&] { fx.ex->execute(p3); }, "plan-seq-discontinuity"));
}

// ---- 6. the mixed-plan deadlock regression. With the host consumer's
//         wait bypassing the single issue order, this exact legal plan
//         deadlocked every time: the worker blocked on the
//         d2h'd read's flag whose setter notify sat behind the parked
//         gates, and the gates parked on host mints behind the blocked
//         worker — circular. One submission order dissolves it.

void scenario_deferral_mixed_repro()
{
    Fx fx;

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(0, 0, kHost, 0), // 0: W's host home
        link(0, 0, kGpu, 0),  // 1: W's h2d destination
        tile(kHostFill11, kHost, {opnd(0, Access::Write, kHost, 0)}), // 2
        copy(0, 0, {kHost, 0}, {kGpu, 0}),                            // 3
        link(1, 0, kHost, 1), // 4: X's host home
        link(1, 0, kGpu, 1),  // 5: X's h2d destination
        tile(kHostFill22, kHost, {opnd(1, Access::Write, kHost, 1)}), // 6
        copy(1, 0, {kHost, 1}, {kGpu, 1}),                            // 7
        link(2, 0, kGpu, 2),  // 8: Y born on the gpu
        tile(kGpuFill33, kGpu, {opnd(2, Access::Write, kGpu, 2)}),   // 9
        link(2, 0, kHost, 2), // 10: Y's host replica
        copy(2, 0, {kGpu, 2}, {kHost, 2}),                            // 11
        tile(kHostCheck33, kHost, {opnd(2, Access::Read, kHost, 2)}), // 12
    };
    fx.ex->execute(p);

    // The host reader saw the device writer's bytes THROUGH the parked
    // gates and the deferred d2h — the plan completes, bytes correct.
    CHECK(fx.host_check_result.load() == 1);
    CHECK(fx.ex->ops_issued() == 13);
    CHECK(fx.ex->last_issue().issue_log.size() == 13);
    CHECK(fx.ex->last_issue().reconcile_ok);
    // The waits' shape: two host-mint gates (3, 7), one device wait
    // (11 on 9), one host-consumer wait (12 on 11).
    OpRef const r2 = make_op_ref(0, 2);
    OpRef const r6 = make_op_ref(0, 6);
    OpRef const r9 = make_op_ref(0, 9);
    OpRef const r11 = make_op_ref(0, 11);
    CHECK(fx.ex->last_issue().issued_waits[3]
          == std::vector<OpRef>{r2});
    CHECK(fx.ex->last_issue().issued_waits[7]
          == std::vector<OpRef>{r6});
    CHECK(fx.ex->last_issue().issued_waits[11]
          == std::vector<OpRef>{r9});
    CHECK(fx.ex->last_issue().issued_waits[12]
          == std::vector<OpRef>{r11});
    CHECK(fx.ex->plans_executed() == 1);
}

// ---- 7. a refusal INSIDE a deferred step: the census at the barrier
//         names the step's exact op (the bridge tag), dependents never
//         run.

void scenario_deferred_refusal()
{
    Fx fx;

    unsigned char *canary = static_cast<unsigned char *>(fx.gpu.addr(2));
    CHECK(cudaMemset(canary, 0, chunk_class_bytes()) == cudaSuccess);
    CHECK(cudaStreamSynchronize(0) == cudaSuccess);

    constexpr TaskId kUnregistered = 78;
    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(0, 0, kHost, 0), // 0
        link(0, 0, kGpu, 1),  // 1
        link(9, 0, kGpu, 2),  // 2: the canary chunk
        tile(kHostFill11, kHost, {opnd(0, Access::Write, kHost, 0)}), // 3
        copy(0, 0, {kHost, 0}, {kGpu, 1}), // 4: the FIRST gate parks
        tile(kUnregistered, kGpu,
            {opnd(9, Access::Write, kGpu, 2)}), // 5: deferred -> refuses
        tile(kGpuFillBB, kGpu, {opnd(9, Access::Write, kGpu, 2)}), // 6
        unlink(9, 0, kGpu, 2),                                     // 7
        unlink(0, 0, kGpu, 1),                                     // 8
        unlink(0, 0, kHost, 0),                                    // 9
    };

    bool threw = false;
    Census caught;
    try
    {
        fx.ex->execute(p);
    }
    catch (ExecutorCensusError const &e)
    {
        threw = true;
        caught = e.census;
    }
    CHECK(threw);
    // The bridge tag names the faulting step's own op: position 5, the
    // unregistered launch — not the barrier's lane-last-op fallback.
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    CHECK(caught.fault.find("refused") != std::string::npos);
    CHECK(caught.fault.find("78") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=5)") !=
        std::string::npos);
    CHECK(caught.op.has_value());
    CHECK(caught.op->u.tile_op.task == kUnregistered);
    CHECK(fx.ex->plans_executed() == 0); // the close never completed

    // Dependents never ran: the deferred fill of position 6 never
    // executed (the fixture drains the lanes itself; the faulted
    // bridge re-throws at the drain — expected, swallowed).
    try
    {
        fx.gpu.barrier();
    }
    catch (std::exception const &)
    {
    }
    std::vector<unsigned char> back(chunk_class_bytes(), 1);
    CHECK(cudaMemcpy(back.data(), canary, chunk_class_bytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    bool any = false;
    for (unsigned char v : back)
    {
        any = any || v != 0;
    }
    CHECK(!any);
}

// ---- 8. the fault-aware release: a host-lane fault behind a parked
//         gate releases the gate into the fault path — the census
//         delivers at the barrier, the deferred canary never runs.

void scenario_host_fault_release()
{
    Fx fx;

    unsigned char *canary = static_cast<unsigned char *>(fx.gpu.addr(2));
    CHECK(cudaMemset(canary, 0, chunk_class_bytes()) == cudaSuccess);
    CHECK(cudaStreamSynchronize(0) == cudaSuccess);

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(9, 0, kHost, 2), // 0: T9's host home (the dependent tile)
        link(9, 0, kGpu, 2),  // 1: T9's h2d destination (the canary)
        tile(kHostFill11, kHost, {opnd(9, Access::Write, kHost, 2)}), // 2
        link(0, 0, kHost, 0), // 3: T0's host home
        link(0, 0, kGpu, 1),  // 4: T0's h2d destination
        tile(kHostFault, kHost, {opnd(0, Access::Write, kHost, 0)}), // 5
        copy(0, 0, {kHost, 0}, {kGpu, 1}), // 6: gates on the dead mint
        copy(9, 0, {kHost, 2}, {kGpu, 2}), // 7: SAME lane's queue, behind
                                           // the gate — the dependent
    };

    bool threw = false;
    Census caught;
    try
    {
        fx.ex->execute(p);
    }
    catch (ExecutorCensusError const &e)
    {
        threw = true;
        caught = e.census;
    }
    CHECK(threw);
    // The gate surfaced through the host fault; the census at the
    // barrier names the gate's consumer op (position 6) and the
    // fault-aware release.
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    // The twin's fault release (the deadlock closure): a host
    // fault releases the recorded-wait park with its census text.
    CHECK(caught.fault.find("HOST-lane fault") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=6)") !=
        std::string::npos);
    CHECK(fx.ex->plans_executed() == 0);

    // Dependents never ran: the copy of position 7 sat BEHIND the
    // fault-released gate in the same lane's queue — nothing past the
    // faulting step executed (the drain re-throws; expected).
    try
    {
        fx.gpu.barrier();
    }
    catch (std::exception const &)
    {
    }
    std::vector<unsigned char> back(chunk_class_bytes(), 1);
    CHECK(cudaMemcpy(back.data(), canary, chunk_class_bytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    bool any = false;
    for (unsigned char v : back)
    {
        any = any || v != 0;
    }
    CHECK(!any);
}

// ---- 9. the device-fault drain-park (the loop's last deviation): a
//         device lane faults asynchronously while a bridge gate is
//         parked and the host worker's own gate waits a notify that
//         sat on the faulted lane — the OLD close (drain first) parked
//         forever; the sync-first close delivers the fault's census at
//         the barrier, releases every park, and the dependents never
//         run. Runs in a FORKED CHILD with alarm() so a regression
//         fails the check instead of hanging ctest (CUDA forbids a
//         fork after init: the child is a fresh exec of this binary
//         before any fixture exists).

void child_device_fault_release()
{
    Fx fx;

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(2, 0, kGpu, 2),  // 0: T2's gpu home
        link(2, 0, kHost, 2), // 1: T2's host replica
        tile(kGpuFaultBad, kGpu, {opnd(2, Access::Write, kGpu, 2)}), // 2
        copy(2, 0, {kGpu, 2}, {kHost, 2}),                           // 3
        tile(kHostCheck33, kHost, {opnd(2, Access::Read, kHost, 2)}), // 4
        link(0, 0, kHost, 0), // 5: W's host home
        link(0, 0, kGpu, 1),  // 6: W's h2d destination
        tile(kHostFill11, kHost, {opnd(0, Access::Write, kHost, 0)}), // 7
        copy(0, 0, {kHost, 0}, {kGpu, 1}), // 8: the BRIDGE gate — its
                                          // flag's notify is a worker
                                          // item BEHIND the worker's
                                          // own parked gate of 4
    };

    bool threw = false;
    Census caught;
    try
    {
        fx.ex->execute(p);
    }
    catch (ExecutorCensusError const &e)
    {
        threw = true;
        caught = e.census;
    }
    CHECK(threw);
    // The census names the FAULTING op — position 2, the faulting
    // TILE_OP on the faulted lane — and the release of the parks.
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    CHECK(caught.fault.find("position 2") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=2)") !=
        std::string::npos);
    // The light close's probe: the FAULTING LANE's own terminal-event
    // join delivered the fault (exec(0) joins first — the common
    // killer), before any trailing lane's drain could park on it.
    CHECK(caught.fault.find("exec(0)") != std::string::npos);
    CHECK(fx.ex->plans_executed() == 0);
    // Dependents never ran: the host read of 4 never executed (its
    // gate released into the fault path, the worker skipped the rest).
    CHECK(fx.host_check_result.load() == 0);
}

void scenario_device_fault_release()
{
    // A fresh exec of this binary (CUDA forbids fork after init),
    // alarm-guarded: a regression hangs the child, SIGALRM kills it,
    // and the check FAILS instead of ctest timing out.
    char self[512];
    ssize_t const n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    CHECK(n > 0);
    if (n <= 0)
    {
        return;
    }
    self[n] = '\0';

    pid_t const pid = fork();
    CHECK(pid >= 0);
    if (pid == 0)
    {
        alarm(60); // the hang guard
        execl(self, self, "--child-device-fault",
            static_cast<char *>(nullptr));
        _exit(127);
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    if (WIFSIGNALED(status))
    {
        std::cerr << "CHECK failed: the device-fault child died by "
                     "signal "
                  << WTERMSIG(status)
                  << " (a hang: the close join parked past its lane's "
                     "sync)\n";
        g_checks_failed += 1;
        return;
    }
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// ---- 10. the byte-exact copy check :
//          a plan COPY whose logical chunk is a TAIL chunk transfers
//          exactly the tile payload's occupied span — the bytes prove
//          it (the block's wasted tail keeps the stale plant), and the
//          armed tracer's byte stamp names the same span.

void scenario_copy_span_bytes()
{
    Fx fx(true); // armed: the trace's byte stamp rides the same plan

    fx.tiles.add(kTailTile, 3 * chunk_class_bytes() / 4);
    fx.tiles.add(kFullTile, chunk_class_bytes());
    std::size_t const tail_span = 3 * chunk_class_bytes() / 4;

    // The host landing planted stale WHOLE class: only the occupied
    // span may move — the wasted tail of the chunk's block behind the
    // payload must keep the stale bytes, or the copy overran.
    std::memset(fx.hpool.addr(1), 0xBB, chunk_class_bytes());

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        link(kTailTile, 0, kGpu, 0),  // 0: the tail tile's gpu home
        link(kTailTile, 0, kHost, 1), // 1: its host landing
        tile(kGpuFillAA, kGpu,
            {opnd(kTailTile, Access::Write, kGpu, 0)}), // 2
        copy(kTailTile, 0, {kGpu, 0}, {kHost, 1}),      // 3: THE span
        unlink(kTailTile, 0, kHost, 1),                 // 4
        unlink(kTailTile, 0, kGpu, 0),                  // 5
    };
    fx.ex->execute(p);

    // The bytes: [0, span) carry the fill, [span, class) the stale
    // plant — the copy moved the span and nothing more.
    auto const *host =
        static_cast<unsigned char const *>(fx.hpool.addr(1));
    for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
    {
        CHECK(host[i] == (i < tail_span ? 0xAA : 0xBB));
    }

    // The trace's byte stamp: the copy's measured record carries the
    // occupied span — the true transferred bytes, never the class
    // constant.
    bool seen = false;
    for (MeasuredTraceRecord const &r : fx.tracer->measured_records())
    {
        if (r.kind == PlanOpKind::Copy)
        {
            CHECK(r.bytes == tail_span);
            seen = true;
        }
    }
    CHECK(seen);
}

// ---- 11. the same check at the channel handoff, witnessed: the
//          recording proxies report the byte count the executor handed
//          each channel — the whole class for a FULL chunk, the payload
//          remainder for a TAIL chunk.

void scenario_copy_span_recorded()
{
    Fx fx(false, true); // the recording proxies over the real channels

    // The byte counts are witnessed at the PER-COPY handoff, so the
    // proxies run the DEFAULT realization (class-sized copy()
    // slices) — the run realization would hand the channel one
    // merged copy_run per plan and the per-chunk counts would live
    // only inside it (scenario_copy_run_merge covers that arm).
    fx.rec_d2h->default_slice = true;

    fx.tiles.add(kTailTile, 3 * chunk_class_bytes() / 4);
    fx.tiles.add(kFullTile, chunk_class_bytes());
    std::size_t const tail_span = 3 * chunk_class_bytes() / 4;

    // Plan 1: a FULL chunk's copy — the recorded count is the whole
    // class, the full chunk's occupied span.
    Plan p1;
    p1.plan_seq = 0;
    p1.queue = {
        link(kFullTile, 0, kGpu, 0),
        link(kFullTile, 0, kHost, 1),
        tile(kGpuFillAA, kGpu,
            {opnd(kFullTile, Access::Write, kGpu, 0)}),
        copy(kFullTile, 0, {kGpu, 0}, {kHost, 1}),
        unlink(kFullTile, 0, kHost, 1),
        unlink(kFullTile, 0, kGpu, 0),
    };
    fx.ex->execute(p1);
    CHECK(fx.rec_d2h->copies.size() == 1);
    CHECK(fx.rec_d2h->copies[0].nbytes == chunk_class_bytes());

    // Plan 2: a TAIL chunk's copy — the recorded count is the payload
    // remainder, never clipped, never padded to the class.
    Plan p2;
    p2.plan_seq = 1;
    p2.queue = {
        link(kTailTile, 0, kGpu, 2),
        link(kTailTile, 0, kHost, 3),
        tile(kGpuFillAA, kGpu,
            {opnd(kTailTile, Access::Write, kGpu, 2)}),
        copy(kTailTile, 0, {kGpu, 2}, {kHost, 3}),
        unlink(kTailTile, 0, kHost, 3),
        unlink(kTailTile, 0, kGpu, 2),
    };
    fx.ex->execute(p2);
    CHECK(fx.rec_d2h->copies.size() == 2);
    CHECK(fx.rec_d2h->copies[1].nbytes == tail_span);
}

// ---- 12. the COPY-run merge seam : two
//          consecutive same-lane COPY ops whose spans are contiguous
//          on BOTH endpoints realize as ONE copy_run call carrying
//          the SUMMED nbytes; the channel base class's DEFAULT
//          realization (class-sized copy() slices) moves the identical
//          bytes — the bytes prove the equivalence, not the call shape.

void scenario_copy_run_merge()
{
    Fx fx(false, true); // the recording proxies over the real channels

    fx.tiles.add(kTwoTile, 2 * chunk_class_bytes());
    std::size_t const run_span = 2 * chunk_class_bytes();

    // The same plan shape, driven twice: once through the CUDA
    // channel's one-call copy_run, once through the base class's
    // default slicer. Two fills (the registered body fills a[0] only)
    // feed the two consecutive d2h COPYs.
    auto drive = [&fx](std::uint64_t seq) {
        // A stale plant first: only the merged run's exact bytes may
        // land, so a short or over-long transfer reads off it.
        std::memset(fx.hpool.addr(0), 0xBB, chunk_class_bytes());
        std::memset(fx.hpool.addr(1), 0xBB, chunk_class_bytes());
        Plan p;
        p.plan_seq = seq;
        p.queue = {
            link(kTwoTile, 0, kGpu, 0),  // 0
            link(kTwoTile, 1, kGpu, 1),  // 1
            link(kTwoTile, 0, kHost, 0), // 2
            link(kTwoTile, 1, kHost, 1), // 3
            tile(kGpuFillAA, kGpu,
                {opnd_ix(kTwoTile, 0, Access::Write, kGpu, 0)}), // 4
            tile(kGpuFill5A, kGpu,
                {opnd_ix(kTwoTile, 1, Access::Write, kGpu, 1)}), // 5
            copy(kTwoTile, 0, {kGpu, 0}, {kHost, 0}),             // 6
            copy(kTwoTile, 1, {kGpu, 1}, {kHost, 1}),             // 7
            unlink(kTwoTile, 0, kHost, 0),                        // 8
            unlink(kTwoTile, 1, kHost, 1),                        // 9
            unlink(kTwoTile, 0, kGpu, 0),                         // 10
            unlink(kTwoTile, 1, kGpu, 1),                         // 11
        };
        fx.ex->execute(p);
    };

    // Per-chunk allocations: each chunk is its own
    // cudaMalloc/cudaHostAlloc region, and a driver copy may not span
    // two allocations even when address-adjacent (an invalid argument
    // on sm_75/CUDA 12.4) — so the run merge is bounded by
    // the allocation extent (contig_, one class under the segmented
    // pools) and two whole-chunk copies realize as TWO one-class
    // runs. The landed bytes are the ground truth: identical either way.
    drive(0);
    CHECK(fx.rec_d2h->runs.size() == 2);
    CHECK(fx.rec_d2h->runs[0].nbytes == chunk_class_bytes());
    CHECK(fx.rec_d2h->runs[1].nbytes == chunk_class_bytes());
    CHECK(fx.rec_d2h->copies.empty());
    CHECK(fx.scan_eq(fx.hpool.addr(0), 0xAA));
    CHECK(fx.scan_eq(fx.hpool.addr(1), 0x5A));

    // Pass 2 — the DEFAULT slice realization: the same two one-class
    // runs, each realized as one class-sized copy() slice; landed
    // bytes identical.
    fx.rec_d2h->default_slice = true;
    fx.rec_d2h->runs.clear();
    fx.rec_d2h->copies.clear();
    drive(1);
    CHECK(fx.rec_d2h->runs.size() == 2);
    CHECK(fx.rec_d2h->copies.size() == 2);
    CHECK(fx.rec_d2h->copies[0].nbytes == chunk_class_bytes());
    CHECK(fx.rec_d2h->copies[1].nbytes == chunk_class_bytes());
    CHECK(fx.scan_eq(fx.hpool.addr(0), 0xAA));
    CHECK(fx.scan_eq(fx.hpool.addr(1), 0x5A));

    // The two fills' bytes distinguish the chunks, so the pass-2
    // slice ORDER (chunk 0 then chunk 1, the queue order) is pinned
    // by the landed values above; the run's base address is the first
    // member's src (contiguity's anchor).
    CHECK(fx.rec_d2h->runs[0].src == fx.gpu.addr(0));
    CHECK(fx.rec_d2h->runs[0].dst == fx.hpool.addr(0));
}

} // namespace

int main(int argc, char **argv)
{
    if (argc > 1 &&
        std::string(argv[1]) == "--child-device-fault")
    {
        child_device_fault_release();
        return g_checks_failed == 0 ? 0 : 1;
    }
{
    scenario_full_plan();
    scenario_boundary_mismatch();
    scenario_issue_abort();
    scenario_registry_miss();
    scenario_late_record();
    scenario_deferral_mixed_repro();
    scenario_deferred_refusal();
    scenario_host_fault_release();
    scenario_device_fault_release();
    scenario_copy_span_bytes();
    scenario_copy_span_recorded();
    scenario_copy_run_merge();

    if (g_checks_failed != 0)
    {
        std::cerr << "plan_executor: " << g_checks_failed
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "plan_executor: all checks passed\n";
    return 0;
}
}
