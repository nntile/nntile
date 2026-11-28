/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * backend conformance suite of the chunk-graph executor: one function per
 * case over REAL lanes.
 *
 * @file tests/test_backend_conformance.cu
 * The conformance suite (ctest target backend_conformance): each backend
 * property realized over cuda_backend + host_worker directly where the
 * property is a backend one, and through the executor / the public API
 * where it binds the engine plane. The suite's cases:
 *
 *   conf_lane_order_rmw        begin-after-completion, per lane;
 *   conf_mint_capture          capture-at-call-time under deep
 *                              pipelining, every direction this host
 *                              claims — the across-devices leg RUNS
 *                              its cross-context record/wait storm on
 *                              a 2-GPU host, skips loudly on one;
 *   conf_epoch_reuse           an old-epoch reader in
 *                              flight over an UNLINK/LINK/overwrite;
 *   conf_host_boundary_h2d     a device consumer waits
 *                              a host mint — the DeferralBridge's real
 *                              path;
 *   conf_host_boundary_d2h     a host consumer waits a
 *                              device mint — the completion-flag gate;
 *   conf_no_readback           a static source audit of the engine
 *                              plane's issue path (the four forbidden
 *                              query APIs), the source tree injected
 *                              as NNHAUL_TEST_SOURCE_DIR;
 *   conf_address_stability     addr() valid and stable for the
 *                              run's life, across churn;
 *   conf_whole_chunk_copy      byte-exact chunk_class_bytes(), tail
 *                              included, every claimed pair — the
 *                              device<->device leg runs a
 *                              CudaPeerChannel copy per probed
 *                              direction on a 2-GPU host;
 *   conf_bounce_buffer         DMA-stability: neighbor canaries
 *                              untouched, no hidden staging;
 *   conf_submission_latency    enqueue cost bounded under load;
 *   conf_barrier_completeness  every fired op observed at the join;
 *   conf_fault_propagation     the fault delivered at the barrier,
 *                              no silent mint, dependents never run
 *                              (the device-fault release path, a
 *                              forked alarm-guarded child so a
 *                              regression fails, not hangs);
 *   conf_stamp_fidelity        end-minus-start brackets
 *                              the work, stamps gate nothing;
 *   conf_planned_issued        both halves over N lanes —
 *                              the PlanIssue witness over the four
 *                              lanes of the classic reduction, then a
 *                              full engine-driven batch through
 *                              signal_graph_end/wait with the tracer
 *                              armed (test_hooks' ManagerStats).
 *
 * The canary discipline holds throughout: the HARNESS plants
 * bytes and reads them back strictly at barriers (the harness may
 * touch bytes; the engine never does). The 2-GPU legs (the peer-copy
 * and across-devices mint-capture directions) run their real bodies
 * on a host with two CUDA devices and a probed peer pair; on a
 * 1-GPU host — or for a direction the probe refuses — they print
 * "[backend_conformance] SKIP <leg>: needs 2 CUDA devices" (or the
 * per-direction "pair absent from the probe") and pass. The suite
 * initializes CUDA — run it under the suite's GPU lock.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/binding_table.hh"
#include "../src/cuda_backend.hh"
#include "../src/host_pool.hh"
#include "../src/plan_executor.hh"

// plan.hh now includes the public header for Access, resolving the
// earlier header clash; the standalone CHECK macro
// stands here anyway, same shape as test_plan_executor.cu's.
#include "nnhaul/defs.h"
#include "nnhaul/nnhaul.hh"
#include "nnhaul/test_hooks.hh"

#include <cuda_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef NNHAUL_TEST_SOURCE_DIR
#error "backend_conformance needs NNHAUL_TEST_SOURCE_DIR (set by CMakeLists)"
#endif

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

// ---- the canary kernels (the harness's device-side oracles; the
//      launch fn may enqueue anything on its lane's stream) ----

// The lane-order RMW cell, device side: arr[0] IS the phase, arr[1]
// the poison flag, arr[2 + phase] the per-phase witness. One thread —
// the cell is the serialization witness, not a benchmark.
__global__ void k_rmw_cell(unsigned *arr, unsigned limit)
{
    unsigned const phase = arr[0];
    if (phase >= limit)
    {
        arr[1] = 1u; // a chain longer than the script: loud poison
        return;
    }
    arr[2 + phase] = 1u; // this op's phase was observed
    __threadfence();
    arr[0] = phase + 1u; // the next op's precondition
}

// Whole-chunk byte compare against one byte value; *ok is planted at 1
// and zeroed on any mismatch (a read that observed the producer's
// bytes through the derived syncs leaves it 1).
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

// The epoch-reuse reader: spins a known wall of device time (the
// "reader in flight"), then proves it read the OLD epoch's bytes —
// begin-after-completion made observable by the bytes surviving.
__global__ void k_epoch_reader(
    unsigned char const *p, unsigned char expected, std::size_t n, int *ok,
    long long spin_cycles)
{
    long long const until = clock64() + spin_cycles;
    while (clock64() < until)
    {
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        if (p[i] != expected)
        {
            *ok = 0;
        }
    }
}

// A deliberately slow device op for stamp fidelity: spins, then writes
// a magic word into the operand's first bytes.
__global__ void k_spin_magic(unsigned *cell, long long spin_cycles)
{
    long long const until = clock64() + spin_cycles;
    while (clock64() < until)
    {
    }
    cell[0] = 0xC0FFEEu;
}

// The deferral-trail witness: captures the first and last bytes of the
// chunk the deferred copy just landed (the endpoints prove the WHOLE
// class moved — a torn or stale copy shows in them).
__global__ void k_capture_ends(unsigned char const *p, std::size_t n,
    unsigned char *snap)
{
    snap[0] = p[0];
    snap[n - 1] = p[n - 1];
}

__global__ void k_noop()
{
}

// The device-fault canary: spins ~a few hundred ms, then commits an
// illegal access — the whole plan issues while the kernel still RUNS,
// healthy; the close-join sync then delivers the fault
// deterministically instead of racing the sticky error.
__global__ void k_fault_delayed(int *p)
{
    long long const until = clock64() + 600000000LL;
    while (clock64() < until)
    {
    }
    *p = 1; // p == nullptr: the illegal-address fault
}

// ---- the fixture: the classic reduction's four lanes  ----

constexpr NodeId kGpu = 0;
constexpr NodeId kHost = 1;
constexpr LaneId kLaneExecHost = 1; // exec(host); a gate's producer key
constexpr LaneId kLaneExecGpu = 0; // exec(gpu)
constexpr LaneId kLaneChanG2H = 2; // chan(gpu,host)
constexpr LaneId kLaneChanH2G = 3; // chan(host,gpu): the deferred tail's lane

// The fixed canary task symbols (scalar facts are registration-time
// facts of the symbol; per-pair symbols are registered per case).
constexpr TaskId kGpuRmw = 100;
constexpr TaskId kGpuFill77 = 101;
constexpr TaskId kGpuFill33 = 102;
constexpr TaskId kGpuCheck11 = 103;
constexpr TaskId kGpuCapture = 104;
constexpr TaskId kGpuSlow = 105;
constexpr TaskId kGpuNoop = 106;
constexpr TaskId kGpuFaultBad = 107;
constexpr TaskId kGpuEpoch = 108;
constexpr TaskId kGpuSpinShort = 109; // ~15 ms: a pending head that
                                      // keeps the submission bound's
                                      // driver-queue exposure small
constexpr TaskId kGpuFillBase = 200;  // + pair index
constexpr TaskId kGpuCheckBase = 300; // + pair index
constexpr TaskId kHostRmw = 600;
constexpr TaskId kHostFill11 = 700;
constexpr TaskId kHostSleepFill55 = 701;
constexpr TaskId kHostSleepFill66 = 702;
constexpr TaskId kHostCheck33 = 800;
constexpr TaskId kHostCheck77 = 801;
constexpr TaskId kHostSlow = 900;
constexpr TaskId kHostNoop = 901;
constexpr TaskId kHostFillBase = 710;  // + pair index
constexpr TaskId kHostCheckBase = 810; // + pair index
constexpr TaskId kUnregistered = 888;

// The device witness cells (planted at 1; kernels zero them on
// mismatch), and their fixed slots.
constexpr std::size_t kWitCheck11 = 0;
constexpr std::size_t kWitEpoch = 1;
// The host witness cells (0 = not run / failed; the host tasks set 1).
constexpr std::size_t kWitHostCheck33 = 0;
constexpr std::size_t kWitHostCheck77 = 1;
constexpr std::size_t kWitHostSlow = 2;

// The hook argument of one host-flag notify: the address is handed
// to an in-flight driver host function, so it lives in a deque (stable
// while notifies fly).
struct FlagArg
{
    HostWorkerLane *host;
    EventHandle event;
};

void set_flag_hook(void *arg) noexcept
{
    auto *a = static_cast<FlagArg *>(arg);
    a->host->set_flag(a->event);
}

// The barrier-completeness fire slot and payload: one slot stamp
// plus one global counter — the harness-planted completion-notification
// callback.
struct FireSlot
{
    std::atomic<std::uint32_t> stamp{0};
    std::uint32_t expect = 0;
};

struct Fire2Arg
{
    FireSlot *slot;
    std::atomic<std::uint64_t> *fires;
};

void fire2_hook(void *arg) noexcept
{
    auto *a = static_cast<Fire2Arg *>(arg);
    a->slot->stamp.store(a->slot->expect, std::memory_order_relaxed);
    a->fires->fetch_add(1u, std::memory_order_relaxed);
}

// The lane-order chain hooks (harness-planted notification callbacks; the
// harness may touch bytes — the engine never does):
// - verify_rmw_set: the h2d chain's step — verify the shared SOURCE
//   equals `expect` over the whole class, then write `next` over it
//   (the read-modify-write); the copy behind the hook carries `next`.
// - verify_rmw_add: the d2h chain's step — verify the shared
//   DESTINATION equals `expect` over the whole class (the copy just
//   landed), then increment byte 0 (the RMW).
struct ChainArg
{
    unsigned char expect;
    unsigned char next;
    unsigned char *bytes;
    std::atomic<std::uint32_t> *count;
    std::atomic<std::uint32_t> *poison;
};

void verify_rmw_set(void *arg) noexcept
{
    auto *a = static_cast<ChainArg *>(arg);
    for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
    {
        if (a->bytes[i] != a->expect)
        {
            a->poison->store(1u, std::memory_order_relaxed);
            return;
        }
    }
    std::memset(a->bytes, a->next, chunk_class_bytes());
    a->count->fetch_add(1u, std::memory_order_relaxed);
}

void verify_rmw_add(void *arg) noexcept
{
    auto *a = static_cast<ChainArg *>(arg);
    for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
    {
        if (a->bytes[i] != a->expect)
        {
            a->poison->store(1u, std::memory_order_relaxed);
            return;
        }
    }
    a->bytes[0] = static_cast<unsigned char>(a->bytes[0] + 1u);
    a->count->fetch_add(1u, std::memory_order_relaxed);
}

// A distinct canary byte per chain/pair index (indices stay < 64, so
// every byte differs from its neighbors').
unsigned char pat_byte(std::size_t i)
{
    return static_cast<unsigned char>((i * 37u + 11u) & 0xFFu);
}

bool host_chunk_eq(void const *p, unsigned char byte)
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

// The fixture: one GPU exec lane, the host worker lane, both host
// channels, the deferral bridge wired into every CUDA lane (the
// engine's own wiring — every CUDA barrier drains it first).
struct ConfFx
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
    // registration both read it). The executor-issued plans
    // of this suite COPY tiles 0, 1, 2 and 9, every one a full-class
    // tile here; the span-exact tail case is the plan_executor
    // suite's.
    TileTable tiles;
    std::unique_ptr<PlanTrace> tracer;
    std::unique_ptr<PlanExecutor> ex;
    cudaStream_t mint_stream = nullptr; // the twins' record stream

    std::size_t const chunks;
    int *d_wit = nullptr;             // device witness cells, planted 1
    std::vector<int> host_wit;        // host witness cells
    std::deque<FlagArg> flag_args;    // stable while notifies fly
    std::deque<std::vector<void *>>
        host_args; // the host lane copies the POINTER, not the array:
                   // every operand array handed
                   // to a host launch lives here until the barrier
                   // that retires it — deque references are stable
    std::unordered_map<TaskId, std::function<void(void *const *)>>
        host_tasks; // registered before any host item exists

    std::size_t rmw_limit = 24; // the RMW chain length (the kRmw limit)

    ConfFx(std::size_t chunks_, std::size_t provision,
        std::size_t witness_cells = 8) :
        inv(1, {chunks_, chunks_}, {0}),
        pool(0, provision),
        gpu(0, 0, chunks_, pool),
        host(provision,
            [this](TaskId t, void *const *addrs) {
                auto it = host_tasks.find(t);
                if (it == host_tasks.end())
                {
                    throw std::runtime_error(
                        "conformance host resolver: unknown task (a loud "
                        "lane fault)");
                }
                it->second(addrs);
            }),
        bridge(host),
        hpool(0, chunks_), // unlimited, preallocated
        d2h(0, HostChannelDirection::DeviceToHost, pool),
        h2d(0, HostChannelDirection::HostToDevice, pool),
        chunks(chunks_),
        host_wit(64, 0)
    {
        // The tile rows : one registration per tile the
        // suite's plans COPY, each a full-class payload.
        tiles.add(0, chunk_class_bytes());
        tiles.add(1, chunk_class_bytes());
        tiles.add(2, chunk_class_bytes());
        tiles.add(9, chunk_class_bytes());

        // ---- the fixed device bodies ----
        gpu.register_task(kGpuRmw,
            [this](cudaStream_t s, void *const *a) {
                k_rmw_cell<<<1, 1, 0, s>>>(
                    static_cast<unsigned *>(a[0]),
                    static_cast<unsigned>(rmw_limit));
            });
        gpu.register_task(kGpuFill77,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0x77, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuFill33,
            [](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], 0x33, chunk_class_bytes(), s);
            });
        gpu.register_task(kGpuCheck11,
            [this](cudaStream_t s, void *const *a) {
                k_check_eq<<<1, 256, 0, s>>>(
                    static_cast<unsigned char const *>(a[0]), 0x11,
                    chunk_class_bytes(), d_wit + kWitCheck11);
            });
        gpu.register_task(kGpuCapture,
            [](cudaStream_t s, void *const *a) {
                k_capture_ends<<<1, 1, 0, s>>>(
                    static_cast<unsigned char const *>(a[0]),
                    chunk_class_bytes(),
                    static_cast<unsigned char *>(a[1]));
            });
        gpu.register_task(kGpuSlow,
            [](cudaStream_t s, void *const *a) {
                k_spin_magic<<<1, 1, 0, s>>>(
                    static_cast<unsigned *>(a[0]), 80000000LL);
            });
        gpu.register_task(kGpuNoop, [](cudaStream_t s, void *const *) {
            k_noop<<<1, 1, 0, s>>>();
        });
        gpu.register_task(kGpuFaultBad,
            [](cudaStream_t s, void *const *) {
                k_fault_delayed<<<1, 1, 0, s>>>(nullptr);
            });
        gpu.register_task(kGpuEpoch,
            [this](cudaStream_t s, void *const *a) {
                k_epoch_reader<<<1, 1, 0, s>>>(
                    static_cast<unsigned char const *>(a[0]), 0xE7,
                    chunk_class_bytes(), d_wit + kWitEpoch, 200000000LL);
            });
        gpu.register_task(kGpuSpinShort,
            [](cudaStream_t s, void *const *a) {
                k_spin_magic<<<1, 1, 0, s>>>(
                    static_cast<unsigned *>(a[0]), 30000000LL);
            });

        // ---- the fixed host bodies ----
        host_tasks.reserve(32);
        host_tasks[kHostRmw] = [this](void *const *a) {
            auto *arr = static_cast<unsigned *>(a[0]);
            unsigned const phase = arr[0];
            if (phase >= rmw_limit)
            {
                arr[1] = 1u;
                return;
            }
            arr[2 + phase] = 1u;
            arr[0] = phase + 1u;
        };
        host_tasks[kHostFill11] = [](void *const *a) {
            std::memset(a[0], 0x11, chunk_class_bytes());
        };
        host_tasks[kHostSleepFill55] = [](void *const *a) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::memset(a[0], 0x55, chunk_class_bytes());
        };
        host_tasks[kHostSleepFill66] = [](void *const *a) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::memset(a[0], 0x66, chunk_class_bytes());
        };
        host_tasks[kHostCheck33] = [this](void *const *a) {
            host_wit[kWitHostCheck33] =
                host_chunk_eq(a[0], 0x33) ? 1 : 0;
        };
        host_tasks[kHostCheck77] = [this](void *const *a) {
            host_wit[kWitHostCheck77] =
                host_chunk_eq(a[0], 0x77) ? 1 : 0;
        };
        host_tasks[kHostSlow] = [this](void *const *a) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            host_wit[kWitHostSlow] =
                (static_cast<unsigned char const *>(a[0])[0] == 0x77)
                    ? 1
                    : 0;
        };
        host_tasks[kHostNoop] = [](void *const *) {
        };

        // The device witness cells: planted at 1 (the ok convention).
        if (cudaSetDevice(0) != cudaSuccess ||
            cudaMalloc(&d_wit, witness_cells * sizeof(int)) !=
                cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: the witness cells' setup");
        }
        std::vector<int> ones(witness_cells, 1);
        if (cudaMemcpy(d_wit, ones.data(), witness_cells * sizeof(int),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: the witness cells' plant");
        }
    }

    ~ConfFx()
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
        if (d_wit != nullptr)
        {
            cudaFree(d_wit);
        }
    }

    ConfFx(ConfFx const &) = delete;
    ConfFx &operator=(ConfFx const &) = delete;

    // (Re)builds the executor over the lanes; with_tracer arms the
    // stamps .
    void build_executor(bool with_tracer)
    {
        if (with_tracer)
        {
            tracer = std::make_unique<PlanTrace>(inv, tiles);
            tracer->arm("/tmp/nnhaul-backend-conf-trace.json");
        }
        std::vector<ExecLane *> execs{&gpu, &host};
        std::vector<TransferChannel *> chans{&d2h, &h2d};
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

    // The queue-close barrier order: device lanes first, the host lane
    // last (the barrier-order case) — each CUDA barrier drains the
    // bridge before its sync.
    void join_all()
    {
        // The per-lane drain first (the executor's light-close
        // discipline, fixture-side): a trailing lane's deferred tail
        // lands before the plain joins observe the streams.
        for (LaneId lane : {kLaneExecGpu, kLaneChanG2H, kLaneChanH2G})
        {
            if (bridge.lane_trailing(lane))
            {
                bridge.drain_lane(lane);
            }
        }
        gpu.barrier();
        d2h.barrier();
        h2d.barrier();
        host.barrier();
    }

    void launch_host(TaskId t, std::vector<void *> const &addrs)
    {
        // The enqueue copies the POINTER, not the array: the array
        // must stay stable until the item is served, so it moves into
        // the fixture's deque, paid by the harness the same way the
        // executor pays it. Built then
        // moved: GCC 14's -Warray-bounds overreaches on the inlined
        // one-element vector copy this file's -Werror build rejects.
        std::vector<void *> stable(addrs);
        host_args.push_back(std::move(stable));
        host.launch(t, kGpu, host_args.back().data());
    }

    void launch_gpu(TaskId t, std::vector<void *> const &addrs)
    {
        gpu.launch(t, kGpu, addrs.data());
    }

    int read_wit(std::size_t slot)
    {
        int v = -1;
        if (cudaMemcpy(&v, d_wit + slot, sizeof(int),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: a witness cell's read");
        }
        return v;
    }

    void read_chunk(
        std::vector<unsigned char> &dst, void const *dev_chunk)
    {
        dst.assign(chunk_class_bytes(), 0);
        if (cudaMemcpy(dst.data(), dev_chunk, chunk_class_bytes(),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: a chunk's read-back");
        }
    }

    void plant_dev(void *dev_chunk, unsigned char byte)
    {
        if (cudaMemset(dev_chunk, byte, chunk_class_bytes()) != cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: a device chunk's plant");
        }
        // A sync cudaMemset's host-block does not order against the
        // channels' NON-BLOCKING streams under device contention: on
        // a busy shared GPU the deferred memset can execute after a
        // copy enqueued later on a non-blocking stream, past a clean
        // stream sync (the CI flake of run 41464137985). Join the
        // legacy stream the memset runs on before the caller
        // enqueues channel work that must land after the plant.
        if (cudaStreamSynchronize(0) != cudaSuccess)
        {
            throw std::runtime_error(
                "conformance fixture: a device chunk's plant join");
        }
    }

    void enqueue_flag_notify(ExecLane &lane, EventHandle e)
    {
        flag_args.push_back(FlagArg{&host, e});
        lane.notify(&set_flag_hook, &flag_args.back());
    }
};

// ---- op factories (the plan-carrying cases) ----

PlanOp plink(TileId t, ChunkIndex i, NodeId n, ChunkId c)
{
    return make_link(t, i, n, c);
}

PlanOp punlink(TileId t, ChunkIndex i, NodeId n, ChunkId c)
{
    return make_unlink(t, i, n, c);
}

PlanOp pcopy(TileId t, ChunkIndex i, ChunkLoc s, ChunkLoc d)
{
    return make_copy(t, i, s, d);
}

PlanOp ptile(TaskId task, NodeId n, std::vector<Operand> ops)
{
    return make_tile_op(task, n, std::move(ops));
}

Operand popnd(TileId t, Access a, NodeId n, ChunkId c)
{
    Operand o;
    o.tile = t;
    o.index = 0;
    o.access = a;
    o.node = n;
    o.chunk = c;
    return o;
}

bool have_two_devices()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        return false;
    }
    return n >= 2;
}

void skip_needs_two_devices(char const *leg)
{
    std::cout << "[backend_conformance] SKIP " << leg
              << ": needs 2 CUDA devices\n";
}

// The 2-GPU legs' init acts, in miniature, paid by the harness: both
// primary contexts made real (cudaFree(0) is the documented
// context-initializing no-op), the peer probe (the run's ONLY device
// introspection), and the post-context
// enable. Returns the probe's k*k row-major bits: bit[a*2+b] set iff
// SOURCE a can access DESTINATION b, recorded faithfully — a pair the
// platform refuses is simply ABSENT (no channel, staging would be
// planned dataflow), and that leg then skips loudly per direction.
std::vector<std::uint8_t> prepare_two_device_peer()
{
    for (int ordinal : {0, 1})
    {
        if (cudaSetDevice(ordinal) != cudaSuccess ||
            cudaFree(0) != cudaSuccess)
        {
            throw std::runtime_error(
                "backend_conformance: a 2-GPU leg's context setup "
                "failed");
        }
    }
    std::vector<int> const ordinals{0, 1};
    std::vector<std::uint8_t> const bits = probe_peer_matrix(ordinals);
    enable_peer_access(ordinals, bits);
    return bits;
}

void skip_pair_absent(char const *leg, int a, int b)
{
    std::cout << "[backend_conformance] SKIP " << leg << " " << a
              << "->" << b << ": pair absent from the probe\n";
}

// ==========================================================================
// Case 4 (built first: pure CPU): conf_no_readback — static audit.
// ==========================================================================

void conf_no_readback()
{
    // The engine plane's ISSUE PATH, file by file — every source that
    // executes while an op issues: the loop itself (plan_executor.cc),
    // the engine's close-and-plan and API entry (engine.cc, api.cc),
    // the derivation pass (dependency.cc), the applied transitions
    // (binding_table.cc), the lane lookups (inventory.cc), and the
    // stamp/fire hooks the loop calls in place (plan_trace.cc). A
    // future poll in ANY of them must fail this audit, not evade it.
    // DELIBERATELY EXCLUDED: src/cuda_backend.cc and src/host_worker.cc
    // — the BACKEND plane, where cudaStreamSynchronize IS the barrier
    // realization and the flag/cv machinery is the carve-out
    // ("backend-internal notifier threads feeding the notification
    // plane are backend mechanism, not engine queries").
    char const *files[] = {"src/plan_executor.cc", "src/engine.cc",
        "src/api.cc", "src/dependency.cc", "src/binding_table.cc",
        "src/inventory.cc", "src/plan_trace.cc"};
    char const *queries[] = {"cudaEventQuery", "cudaStreamQuery",
        "cudaEventElapsedTime", "cudaEventSynchronize"};
    for (char const *rel : files)
    {
        std::string const path =
            std::string(NNHAUL_TEST_SOURCE_DIR) + "/" + rel;
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            std::cerr << "CHECK failed: conf_no_readback cannot read "
                      << path << " (NNHAUL_TEST_SOURCE_DIR is wrong?)\n";
            g_checks_failed += 1;
            continue;
        }
        std::string line;
        std::size_t number = 0;
        while (std::getline(in, line))
        {
            ++number;
            for (char const *q : queries)
            {
                if (line.find(q) != std::string::npos)
                {
                    // A hit anywhere — comment included — is a finding:
                    // the engine plane's issue path stays grep-clean of
                    // the state-query APIs .
                    std::cerr << "CHECK failed: conf_no_readback: "
                              << rel << ":" << number << " names " << q
                              << " — the engine plane never queries, "
                                 "polls, or waits on event or lane "
                                 "state\n";
                    g_checks_failed += 1;
                }
            }
        }
    }
}

// ==========================================================================
// Case 5: conf_address_stability — addr() stable across churn.
// ==========================================================================

void conf_address_stability()
{
    ConfFx fx(8, 64);

    // The snapshot before: every chunk's address on every node, each
    // carrying a distinct planted canary the churn never names.
    std::vector<void *> dev_addr(fx.chunks);
    std::vector<void *> host_addr(fx.chunks);
    for (std::size_t i = 0; i < fx.chunks; ++i)
    {
        dev_addr[i] = fx.gpu.addr(static_cast<ChunkId>(i));
        host_addr[i] = fx.hpool.addr(i);
        fx.plant_dev(dev_addr[i], static_cast<unsigned char>(0xF0u + i));
        std::memset(host_addr[i], static_cast<int>(0xE0u + i),
            chunk_class_bytes());
    }
    // Each chunk is its own whole-class allocation:
    // DISTINCT and stable for the run's life (per-block adjacency is
    // deleted machinery — segmented pools never promise it).
    {
        std::vector<void *> dseen(dev_addr.begin(), dev_addr.end());
        std::vector<void *> hseen(host_addr.begin(), host_addr.end());
        std::sort(dseen.begin(), dseen.end());
        std::sort(hseen.begin(), hseen.end());
        CHECK(std::unique(dseen.begin(), dseen.end()) == dseen.end());
        CHECK(std::unique(hseen.begin(), hseen.end()) == hseen.end());
    }

    // The churn: copies both directions, launches, mints and waits —
    // ops 0..1 only; chunks 2..7 stay untouched canaries.
    std::vector<void *> a0{dev_addr[0]};
    for (int round = 0; round < 3; ++round)
    {
        fx.h2d.copy(host_addr[0], dev_addr[0], chunk_class_bytes());
        fx.launch_gpu(kGpuFill77, a0);
        EventHandle const e = fx.pool.acquire();
        fx.gpu.mint(e);
        fx.d2h.wait(e);
        fx.d2h.copy(dev_addr[0], host_addr[1], chunk_class_bytes());
        fx.pool.release(e);
        for (int i = 0; i < 16; ++i)
        {
            fx.launch_gpu(kGpuNoop, a0);
        }
    }
    fx.join_all();

    // The verification after: every address identical, every untouched
    // chunk's canary intact (the bytes moved nowhere by themselves).
    for (std::size_t i = 0; i < fx.chunks; ++i)
    {
        CHECK(fx.gpu.addr(static_cast<ChunkId>(i)) == dev_addr[i]);
        CHECK(fx.hpool.addr(i) == host_addr[i]);
    }
    std::vector<unsigned char> back(chunk_class_bytes(), 0);
    for (std::size_t i = 2; i < fx.chunks; ++i)
    {
        fx.read_chunk(back, dev_addr[i]);
        CHECK(back[0] == static_cast<unsigned char>(0xF0u + i));
        CHECK(back[chunk_class_bytes() - 1] ==
              static_cast<unsigned char>(0xF0u + i));
        CHECK(host_chunk_eq(host_addr[i],
            static_cast<unsigned char>(0xE0u + i)));
    }
}

// ==========================================================================
// Case 6: conf_whole_chunk_copy — byte-exact copies: the full
// class for a full chunk (tail included, every pair this host claims —
// device<->host both ways; the device<->device leg skips loudly), and
// the span guard's belt: a zero or over-class copy count is a loud
// construction/submission fault, never silently clipped.
// ==========================================================================

void fill_position_dependent(
    std::vector<unsigned char> &buf, unsigned seed)
{
    buf.assign(chunk_class_bytes(), 0);
    for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
    {
        buf[i] = static_cast<unsigned char>(
            (i * 131u + seed * 7u + 5u) & 0xFFu);
    }
}

bool matches_position_dependent(
    std::vector<unsigned char> const &buf, unsigned seed)
{
    if (buf.size() != chunk_class_bytes())
    {
        return false;
    }
    for (std::size_t i = 0; i < chunk_class_bytes(); ++i)
    {
        if (buf[i] !=
            static_cast<unsigned char>((i * 131u + seed * 7u + 5u) & 0xFFu))
        {
            return false;
        }
    }
    return true;
}

// The device <-> device leg, per direction on the probed pair: one
// CudaPeerChannel — its stream on the DESTINATION device, its event
// family the destination's pool  — one whole-class
// cudaMemcpyPeerAsync, byte-exact against a position-dependent
// pattern, the tail byte named. Runs only on a 2-GPU host whose
// probe claimed the direction.
void conf_whole_chunk_copy_peer(std::uint8_t const *bits)
{
    for (int a = 0; a < 2; ++a)
    {
        int const b = 1 - a;
        std::size_t const bit = static_cast<std::size_t>(a) * 2u +
                                static_cast<std::size_t>(b);
        if (bits[bit] == 0u)
        {
            skip_pair_absent("conf_whole_chunk_copy device<->device",
                a, b);
            continue;
        }
        if (cudaSetDevice(a) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy peer: cudaSetDevice(a)");
        }
        void *src = nullptr;
        if (cudaMalloc(&src, chunk_class_bytes()) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy peer: the source chunk");
        }
        CudaEventPool pool(b, 16); // the destination's event family
        CudaPeerChannel chan(a, b, pool);
        void *dst = nullptr;
        if (cudaMalloc(&dst, chunk_class_bytes()) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy peer: the destination chunk");
        }

        unsigned const seed = 100u + static_cast<unsigned>(a);
        std::vector<unsigned char> pattern;
        fill_position_dependent(pattern, seed);
        if (cudaMemcpy(src, pattern.data(), chunk_class_bytes(),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemset(dst, 0xDD, chunk_class_bytes()) != cudaSuccess ||
            // the plants must land before the peer channel's
            // non-blocking copy (plant_dev's comment): join them
            cudaStreamSynchronize(0) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy peer: the plants");
        }
        chan.copy(src, dst, chunk_class_bytes()); // one cudaMemcpyPeerAsync, whole class
        chan.barrier();
        std::vector<unsigned char> back(chunk_class_bytes(), 0);
        if (cudaMemcpy(back.data(), dst, chunk_class_bytes(),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy peer: the read-back");
        }
        CHECK(matches_position_dependent(back, seed));
        CHECK(back[chunk_class_bytes() - 1] ==
              static_cast<unsigned char>(
                  ((chunk_class_bytes() - 1) * 131u + seed * 7u + 5u) &
                  0xFFu)); // the tail, named

        cudaSetDevice(b);
        cudaFree(dst);
        cudaSetDevice(a);
        cudaFree(src);
    }
    cudaSetDevice(0);
}

void conf_whole_chunk_copy()
{
    ConfFx fx(6, 64);
    std::vector<unsigned char> pattern(chunk_class_bytes(), 0);
    std::vector<unsigned char> back(chunk_class_bytes(), 0);

    // The span guard (the base chunk-class belt, before any in-flight
    // work exists): a copy of 0 bytes or beyond the one chunk class is
    // a construction/submission fault — loud, never silently clipped
    // to fit the memcpy.
    for (std::size_t bad :
        {std::size_t{0}, chunk_class_bytes() + 1, 2 * chunk_class_bytes()})
    {
        bool threw = false;
        try
        {
            fx.h2d.copy(fx.hpool.addr(0), fx.gpu.addr(5), bad);
        }
        catch (std::exception const &)
        {
            threw = true;
        }
        CHECK(threw);
    }

    // host -> device: two chunks, two position-dependent seeds (the
    // final bytes differ per seed — the TAIL carries information too).
    for (unsigned seed = 1; seed <= 2; ++seed)
    {
        fill_position_dependent(pattern, seed);
        std::memcpy(fx.hpool.addr(seed - 1), pattern.data(),
            chunk_class_bytes());
        fx.plant_dev(fx.gpu.addr(seed - 1), 0xDD); // the stale plant
        fx.h2d.copy(fx.hpool.addr(seed - 1), fx.gpu.addr(seed - 1),
            chunk_class_bytes());
    }
    fx.h2d.barrier();
    for (unsigned seed = 1; seed <= 2; ++seed)
    {
        fx.read_chunk(back, fx.gpu.addr(seed - 1));
        CHECK(matches_position_dependent(back, seed));
        CHECK(back[chunk_class_bytes() - 1] ==
              static_cast<unsigned char>(
                  ((chunk_class_bytes() - 1) * 131u + seed * 7u + 5u) &
                  0xFFu));
    }

    // device -> host: the mirrored leg, fresh chunks.
    for (unsigned seed = 3; seed <= 4; ++seed)
    {
        fill_position_dependent(pattern, seed);
        if (cudaMemcpy(fx.gpu.addr(seed), pattern.data(),
                chunk_class_bytes(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error(
                "conf_whole_chunk_copy: the device plant");
        }
        std::memset(fx.hpool.addr(seed), 0xEE, chunk_class_bytes());
        fx.d2h.copy(fx.gpu.addr(seed), fx.hpool.addr(seed), chunk_class_bytes());
    }
    fx.d2h.barrier();
    for (unsigned seed = 3; seed <= 4; ++seed)
    {
        std::vector<unsigned char> const host_back(
            static_cast<unsigned char const *>(fx.hpool.addr(seed)),
            static_cast<unsigned char const *>(fx.hpool.addr(seed)) +
                chunk_class_bytes());
        CHECK(matches_position_dependent(host_back, seed));
        CHECK(host_back[chunk_class_bytes() - 1] ==
              static_cast<unsigned char>(
                  ((chunk_class_bytes() - 1) * 131u + seed * 7u + 5u) &
                  0xFFu));
    }

    // device <-> device: the peer leg on a 2-GPU host, a loud skip on
    // a single-GPU test host.
    if (!have_two_devices())
    {
        skip_needs_two_devices("conf_whole_chunk_copy device<->device");
        return;
    }
    std::vector<std::uint8_t> const bits = prepare_two_device_peer();
    conf_whole_chunk_copy_peer(bits.data());
}

// ==========================================================================
// Case 7: conf_bounce_buffer — DMA-stability: the neighbors of a
// copied chunk keep their canaries; no hidden staging exists.
// ==========================================================================

void conf_bounce_buffer()
{
    ConfFx fx(8, 64);

    // Every chunk on both nodes carries a distinct canary.
    for (std::size_t i = 0; i < 5; ++i)
    {
        fx.plant_dev(fx.gpu.addr(static_cast<ChunkId>(i)),
            static_cast<unsigned char>(0xA0u + i));
        std::memset(fx.hpool.addr(i), static_cast<int>(0xB0u + i),
            chunk_class_bytes());
    }

    // Copy the MIDDLE chunks only: host2 -> dev2 (h2d), dev3 -> host3
    // (d2h), directly over the pools' bytes (no staging named).
    fx.h2d.copy(fx.hpool.addr(2), fx.gpu.addr(2), chunk_class_bytes());
    fx.d2h.copy(fx.gpu.addr(3), fx.hpool.addr(3), chunk_class_bytes());
    fx.join_all();

    // The neighbors are untouched, whole class, tail included.
    std::vector<unsigned char> back(chunk_class_bytes(), 0);
    for (std::size_t i : {std::size_t{0}, std::size_t{1}, std::size_t{4}})
    {
        fx.read_chunk(back, fx.gpu.addr(static_cast<ChunkId>(i)));
        CHECK(host_chunk_eq(back.data(), static_cast<unsigned char>(0xA0u + i)));
        CHECK(host_chunk_eq(fx.hpool.addr(i),
            static_cast<unsigned char>(0xB0u + i)));
    }
    // The copied middles hold exactly the source's canary.
    fx.read_chunk(back, fx.gpu.addr(2));
    CHECK(host_chunk_eq(back.data(), 0xB2));
    CHECK(host_chunk_eq(fx.hpool.addr(3), 0xA3));
}

// ==========================================================================
// Case 1: conf_lane_order_rmw — begin-after-completion: a chain of
// read-modify-write canary ops on EVERY lane (the device exec lane,
// the host lane, both host channels), the final bytes proving
// serialization with memory-ordering strength.
// ==========================================================================

void conf_lane_order_rmw()
{
    constexpr std::size_t kN = 24; // the cell-chain length (exec lanes)
    constexpr std::size_t kM = 6;  // the copy-chain length (channels)
    ConfFx fx(16, 256, 8);
    fx.rmw_limit = kN;

    // ---- the exec(0) chain: kN RMW kernels over one device cell (a
    //      chunk OUTSIDE both channels' ranges — the chains must not
    //      share bytes with each other, only their own) ----
    unsigned char *const cell_g =
        static_cast<unsigned char *>(fx.gpu.addr(12));
    {
        std::vector<unsigned char> zeros(chunk_class_bytes(), 0);
        if (cudaMemcpy(cell_g, zeros.data(), chunk_class_bytes(),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("conf_lane_order_rmw: the cell plant");
        }
    }
    std::vector<void *> cell_args{cell_g};

    // ---- the exec(host) chain: kN RMW resolver calls over pinned bytes
    unsigned char *const cell_h =
        static_cast<unsigned char *>(fx.hpool.addr(10));
    std::memset(cell_h, 0, chunk_class_bytes());

    // ---- the chan(1,0) (h2d) chain: hook_i verifies the shared source
    //      == pattern_i (the previous hook's write; the plant for i=0),
    //      RMWs it to pattern_{i+1}, and the copy behind the hook
    //      carries it into its own device chunk ----
    unsigned char *const h2d_src = static_cast<unsigned char *>(fx.hpool.addr(11));
    std::memset(h2d_src, pat_byte(0), chunk_class_bytes()); // hook 0's plant
    std::atomic<std::uint32_t> h2d_count{0};
    std::atomic<std::uint32_t> h2d_poison{0};
    std::deque<ChainArg> h2d_args;
    std::vector<void *> h2d_dst(kM);
    for (std::size_t i = 0; i < kM; ++i)
    {
        h2d_dst[i] = fx.gpu.addr(static_cast<ChunkId>(i));
        fx.plant_dev(h2d_dst[i], 0xDD); // the stale plant
    }

    // ---- the chan(0,1) (d2h) chain: copy_i lands pattern_i (planted
    //      per source chunk) into the SHARED host destination, then
    //      hook_i verifies the whole class and RMWs byte 0 ----
    unsigned char *const d2h_dst = static_cast<unsigned char *>(fx.hpool.addr(12));
    std::memset(d2h_dst, 0xEE, chunk_class_bytes()); // the stale plant
    std::atomic<std::uint32_t> d2h_count{0};
    std::atomic<std::uint32_t> d2h_poison{0};
    std::deque<ChainArg> d2h_args;
    std::vector<void *> d2h_src(kM);
    for (std::size_t i = 0; i < kM; ++i)
    {
        d2h_src[i] = fx.gpu.addr(static_cast<ChunkId>(kM + i));
        fx.plant_dev(d2h_src[i], pat_byte(i));
    }

    // The chains, interleaved lane by lane — every lane loaded at once,
    // each chain independent (its own chunks), so only the lane's own
    // order can carry the bytes.
    for (std::size_t i = 0; i < kN || i < kM; ++i)
    {
        if (i < kN)
        {
            fx.launch_gpu(kGpuRmw, cell_args);       // exec(0)
            fx.launch_host(kHostRmw, {cell_h});      // exec(host)
        }
        if (i < kM)
        {
            // chan(1,0): hook RMWs the source, the copy carries it.
            h2d_args.push_back(ChainArg{pat_byte(i), pat_byte(i + 1),
                h2d_src, &h2d_count, &h2d_poison});
            fx.h2d.notify(&verify_rmw_set, &h2d_args.back());
            fx.h2d.copy(h2d_src, h2d_dst[i], chunk_class_bytes());
            // chan(0,1): the copy lands, the hook verifies and RMWs.
            fx.d2h.copy(d2h_src[i], d2h_dst, chunk_class_bytes());
            d2h_args.push_back(ChainArg{pat_byte(i), 0, d2h_dst,
                &d2h_count, &d2h_poison});
            fx.d2h.notify(&verify_rmw_add, &d2h_args.back());
        }
    }
    fx.join_all();

    // exec(0)'s final bytes: the phase advanced exactly kN steps, every
    // phase witnessed, no poison — each op read the previous op's
    // write (begin-after-completion, not merely enqueue order).
    {
        std::vector<unsigned char> back(chunk_class_bytes(), 0);
        fx.read_chunk(back, cell_g);
        auto word_at = [&back](std::size_t w) {
            unsigned v = 0;
            std::memcpy(&v, back.data() + w * sizeof(unsigned),
                sizeof(unsigned));
            return v;
        };
        CHECK(word_at(0) == kN);
        CHECK(word_at(1) == 0);
        bool all = true;
        for (std::size_t p = 0; p < kN; ++p)
        {
            all = all && word_at(2 + p) == 1u;
        }
        CHECK(all);
    }
    // exec(host)'s final bytes: the same proof in pinned memory.
    {
        auto const *arr =
            reinterpret_cast<unsigned const *>(cell_h);
        CHECK(arr[0] == kN);
        CHECK(arr[1] == 0);
        bool all = true;
        for (std::size_t p = 0; p < kN; ++p)
        {
            all = all && arr[2 + p] == 1u;
        }
        CHECK(all);
    }
    // chan(1,0)'s final bytes: every device chunk carries the hook
    // that preceded its copy (pattern_{i+1}); the source's own RMW
    // chain ran every step; no poison.
    CHECK(h2d_count.load() == kM);
    CHECK(h2d_poison.load() == 0);
    CHECK(host_chunk_eq(h2d_src, pat_byte(kM)));
    {
        std::vector<unsigned char> back(chunk_class_bytes(), 0);
        for (std::size_t i = 0; i < kM; ++i)
        {
            fx.read_chunk(back, h2d_dst[i]);
            CHECK(host_chunk_eq(back.data(), pat_byte(i + 1)));
        }
    }
    // chan(0,1)'s final bytes: the shared destination holds the LAST
    // copy's pattern with byte 0 incremented by the LAST hook's RMW.
    CHECK(d2h_count.load() == kM);
    CHECK(d2h_poison.load() == 0);
    {
        unsigned char const expected0 = static_cast<unsigned char>(
            pat_byte(kM - 1) + 1u);
        CHECK(d2h_dst[0] == expected0);
        bool all = true;
        for (std::size_t i = 1; i < chunk_class_bytes(); ++i)
        {
            all = all && d2h_dst[i] == pat_byte(kM - 1);
        }
        CHECK(all);
    }
}

// ==========================================================================
// Case 3: conf_epoch_reuse — an old-epoch reader in flight over an
// UNLINK/LINK/overwrite — bytes
// intact. The reader spins on the device while the harness relinks
// the slot in the binding table (silence — no lane call exists)
// and the new-epoch writer's copy waits the reader's mint.
// ==========================================================================

void conf_epoch_reuse()
{
    ConfFx fx(8, 64);

    unsigned char *const chunk = static_cast<unsigned char *>(fx.gpu.addr(4));
    unsigned char *const fresh = static_cast<unsigned char *>(fx.hpool.addr(4));
    fx.plant_dev(chunk, 0xE7);   // the OLD epoch's bytes
    std::memset(fresh, 0x99, chunk_class_bytes()); // the NEW epoch's bytes

    // The binding-table churn surface (engine memory, no lane): the
    // old epoch linked, then UNLINK/LINK over the in-flight reader.
    BindingTable table({fx.chunks, fx.chunks});
    table.link(kGpu, 4, LogicalChunk{7, 0});
    table.actualize(kGpu, 4);
    CHECK(table.free_count(kGpu) == fx.chunks - 1);

    // The old-epoch reader launches (in flight for ~100 ms of device
    // time), its mint follows, the harness relinks the slot — no
    // backend call names a binding change — and the
    // new-epoch writer's copy waits the reader's mint before it lands.
    std::vector<void *> a{chunk};
    fx.launch_gpu(kGpuEpoch, a);
    EventHandle const e = fx.pool.acquire();
    fx.gpu.mint(e);
    table.unlink(kGpu, 4);                 // the epoch dies
    CHECK(table.free_count(kGpu) == fx.chunks);
    table.link(kGpu, 4, LogicalChunk{8, 0}); // the fresh epoch, outdated
    CHECK(table.free_count(kGpu) == fx.chunks - 1);
    fx.h2d.wait(e); // the carried USER fact's edge, issued by hand
    fx.h2d.copy(fresh, chunk, chunk_class_bytes());
    fx.join_all();

    // The reader's bytes were intact (it read the OLD epoch whole —
    // the overwrite began only after its completion), and the NEW
    // epoch's bytes are what the slot now holds.
    CHECK(fx.read_wit(kWitEpoch) == 1);
    std::vector<unsigned char> back(chunk_class_bytes(), 0);
    fx.read_chunk(back, chunk);
    CHECK(host_chunk_eq(back.data(), 0x99));
    fx.pool.release(e);
}

// ==========================================================================
// Case 3b: conf_host_boundary_h2d — the device-consumer direction: a
// device lane waits a HOST-lane mint by DEFERRED ENQUEUE through the
// backend-side DeferralBridge (the mechanism the engine chose; the gate host
// function stays landed and unused: the driver wedge). The deferred tail
// trails host progress.
// ==========================================================================

void conf_host_boundary_h2d()
{
    ConfFx fx(8, 256);

    unsigned char *const src1 = static_cast<unsigned char *>(fx.hpool.addr(0));
    unsigned char *const src2 = static_cast<unsigned char *>(fx.hpool.addr(1));
    void *const dst = fx.gpu.addr(0);
    void *const snap = fx.gpu.addr(1);
    std::memset(src1, 0xEE, chunk_class_bytes()); // stale plants
    std::memset(src2, 0xEE, chunk_class_bytes());
    fx.plant_dev(dst, 0xDD);
    {
        std::vector<unsigned char> zeros(chunk_class_bytes(), 0);
        if (cudaMemcpy(snap, zeros.data(), chunk_class_bytes(),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("conf_host_boundary_h2d: snap plant");
        }
    }

    // Producer 1 (deliberately slow on the host lane), its mint, and
    // the DEFERRED tail: the gate parks the bridge until the mint is
    // host-visible, then the copy and the capture ride the FIFO.
    fx.launch_host(kHostSleepFill55, {src1});
    EventHandle const h1 = fx.host.pool().acquire();
    fx.host.mint(h1);
    fx.bridge.gate(kLaneChanH2G, make_op_ref(0, 0), h1, kLaneExecHost);
    fx.bridge.submit(kLaneChanH2G, make_op_ref(0, 0), [&fx, src1, dst] {
        fx.h2d.copy(src1, dst, chunk_class_bytes());
    });
    std::vector<void *> cap_args{dst, snap};
    fx.bridge.submit(kLaneChanH2G, make_op_ref(0, 1), [&fx, cap_args] {
        fx.gpu.launch(kGpuCapture, kGpu, cap_args.data());
    });

    // Producer 2 overwrites the same device destination through its
    // own gate — the single FIFO order keeps the tails from crossing.
    fx.launch_host(kHostSleepFill66, {src2});
    EventHandle const h2 = fx.host.pool().acquire();
    fx.host.mint(h2);
    fx.bridge.gate(kLaneChanH2G, make_op_ref(0, 2), h2, kLaneExecHost);
    fx.bridge.submit(kLaneChanH2G, make_op_ref(0, 2), [&fx, src2, dst] {
        fx.h2d.copy(src2, dst, chunk_class_bytes());
    });

    fx.join_all(); // device lanes first: the drain joins the deferred
                   // tail, then the host lane drains
    CHECK(fx.bridge.idle()); // the close belt: nothing left parked

    // The capture observed producer 1's bytes (the deferred copy
    // trailed host progress), and the destination now holds producer
    // 2's — both ends of the class, so the whole chunk moved.
    std::vector<unsigned char> back(chunk_class_bytes(), 0);
    fx.read_chunk(back, snap);
    CHECK(back[0] == 0x55);
    CHECK(back[chunk_class_bytes() - 1] == 0x55);
    fx.read_chunk(back, dst);
    CHECK(host_chunk_eq(back.data(), 0x66));
    fx.host.pool().release(h1);
    fx.host.pool().release(h2);
}

// ==========================================================================
// Case 3c: conf_host_boundary_d2h — the host-consumer direction: the
// host lane waits a DEVICE-lane mint by gating the dependent TILE_OP on
// the mint's completion notification (a flag set by a lane-position host
// callback — never a poll). Both waves of a
// two-deep chain prove the gate per mint.
// ==========================================================================

void conf_host_boundary_d2h()
{
    ConfFx fx(8, 256);

    void *const dev = fx.gpu.addr(2);
    unsigned char *const host0 = static_cast<unsigned char *>(fx.hpool.addr(2));
    unsigned char *const host1 = static_cast<unsigned char *>(fx.hpool.addr(3));
    std::memset(host0, 0xEE, chunk_class_bytes()); // stale plants
    std::memset(host1, 0xEE, chunk_class_bytes());

    // Wave 1: device fill (0x77) on the wave's OWN source chunk, the
    // mint right after the work, the d2h copy waits it, the
    // flag-setting notify rides the copy's lane right after ITS mint,
    // and the host consumer gates on the flag. (Per-wave chunks: the
    // engine's derivation would owe the second fill a WAR wait on the
    // first copy — this case pins the GATE, so it simply never shares
    // the hazard.)
    std::vector<void *> a{dev};
    fx.launch_gpu(kGpuFill77, a);
    EventHandle const e1 = fx.pool.acquire();
    fx.gpu.mint(e1);
    fx.d2h.wait(e1);
    fx.d2h.copy(dev, host0, chunk_class_bytes());
    EventHandle const e2 = fx.pool.acquire();
    fx.d2h.mint(e2);
    EventHandle const f1 = fx.host.pool().acquire();
    fx.enqueue_flag_notify(fx.d2h, f1);
    fx.host.wait(f1);
    fx.launch_host(kHostCheck77, {host0});

    // Wave 2: the same shape with the second pattern and its own
    // chunk (the wave's fill reads nothing the first wave wrote).
    void *const dev2 = fx.gpu.addr(3);
    std::vector<void *> a2{dev2};
    fx.launch_gpu(kGpuFill33, a2);
    EventHandle const e3 = fx.pool.acquire();
    fx.gpu.mint(e3);
    fx.d2h.wait(e3);
    fx.d2h.copy(dev2, host1, chunk_class_bytes());
    EventHandle const e4 = fx.pool.acquire();
    fx.d2h.mint(e4);
    EventHandle const f2 = fx.host.pool().acquire();
    fx.enqueue_flag_notify(fx.d2h, f2);
    fx.host.wait(f2);
    fx.launch_host(kHostCheck33, {host1});

    fx.join_all();
    CHECK(fx.host_wit[kWitHostCheck77] == 1);
    CHECK(fx.host_wit[kWitHostCheck33] == 1);
    fx.pool.release(e1);
    fx.pool.release(e2);
    fx.pool.release(e3);
    fx.pool.release(e4);
    fx.host.pool().release(f1);
    fx.host.pool().release(f2);
}

// ==========================================================================
// Case 2: conf_mint_capture — under deep pipelining: a wait
// captures the mint's state AT CALL TIME, the record strictly
// preceding . Every direction this host
// claims: within one device (exec -> chan), device -> host consumer
// (the flag gate), host -> device consumer (the deferred enqueue).
// Many mints stay outstanding — no barrier until the whole storm is
// enqueued. The across-devices leg needs 2 GPUs (skips loudly).
// ==========================================================================

constexpr std::size_t kMintPairs = 40;

// Phase A: device producer -> d2h consumer (cross-stream, one device)
// with a host consumer gating on the copy's mint (both host-boundary
// directions of the same storm). Stale plants everywhere: an ungated
// wait (a capture of a never-recorded event) reads stale bytes.
void conf_mint_capture_phase_a(ConfFx &fx)
{
    std::vector<EventHandle> dev_handles;
    std::vector<EventHandle> host_flags;
    dev_handles.reserve(2 * kMintPairs);
    host_flags.reserve(kMintPairs);

    // The per-pair task symbols (scalars are registration-time
    // facts; a varying byte multiplies symbols), registered before any
    // item exists on any lane.
    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        unsigned char const byte = pat_byte(i + 1);
        fx.gpu.register_task(kGpuFillBase + static_cast<TaskId>(i),
            [byte](cudaStream_t s, void *const *a) {
                cudaMemsetAsync(a[0], byte, chunk_class_bytes(), s);
            });
        fx.host_tasks[kHostCheckBase + static_cast<TaskId>(i)] =
            [&fx, i, byte](void *const *a) {
                fx.host_wit[8 + i] = host_chunk_eq(a[0], byte) ? 1 : 0;
            };
    }

    // The stale plants first (the hazard detectors), then one storm of
    // enqueues: fill, mint, wait, copy, mint, flag-notify, gate, check
    // — pair after pair with NO barrier, so dozens of mints stand
    // outstanding while the issue loop keeps appending.
    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        void *const dev = fx.gpu.addr(static_cast<ChunkId>(i));
        unsigned char *const host =
            static_cast<unsigned char *>(fx.hpool.addr(i));
        fx.plant_dev(dev, 0xDD);
        std::memset(host, 0xEE, chunk_class_bytes());
    }
    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        void *const dev = fx.gpu.addr(static_cast<ChunkId>(i));
        unsigned char *const host =
            static_cast<unsigned char *>(fx.hpool.addr(i));
        TaskId const fill = kGpuFillBase + static_cast<TaskId>(i);
        TaskId const check = kHostCheckBase + static_cast<TaskId>(i);

        fx.launch_gpu(fill, {dev}); // the producer's device work
        EventHandle const e1 = fx.pool.acquire();
        fx.gpu.mint(e1); // the record — strictly before the wait below
        dev_handles.push_back(e1);

        fx.d2h.wait(e1); // the capture, at its call time
        fx.d2h.copy(dev, host, chunk_class_bytes());
        EventHandle const e2 = fx.pool.acquire();
        fx.d2h.mint(e2); // the copy's own mint
        dev_handles.push_back(e2);

        EventHandle const f = fx.host.pool().acquire();
        host_flags.push_back(f);
        fx.enqueue_flag_notify(fx.d2h, f); // sets the host flag
        fx.host.wait(f);  // the host consumer's gate
        fx.launch_host(check, {host}); // reads strictly the copied bytes
    }
    fx.join_all();

    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        CHECK(fx.host_wit[8 + i] == 1); // every gated read saw its bytes
    }
    for (EventHandle h : dev_handles)
    {
        fx.pool.release(h);
    }
    for (EventHandle h : host_flags)
    {
        fx.host.pool().release(h);
    }
}

// Phase B: host producer -> device consumer — the DEFERRED-ENQUEUE
// path: every wait and every enqueue of the pair rides the bridge's
// single FIFO in issue order, so the mint's record call strictly
// precedes the consumer's wait call even from the waiter thread.
void conf_mint_capture_phase_b(ConfFx &fx)
{
    std::vector<EventHandle> dev_handles;
    std::vector<EventHandle> host_flags;

    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        unsigned char const byte = pat_byte(i + 65);
        fx.host_tasks[kHostFillBase + static_cast<TaskId>(i)] =
            [byte](void *const *a) {
                std::memset(a[0], byte, chunk_class_bytes());
            };
        int *const wit = fx.d_wit + static_cast<int>(16 + i);
        fx.gpu.register_task(kGpuCheckBase + static_cast<TaskId>(i),
            [byte, wit](cudaStream_t s, void *const *a) {
                k_check_eq<<<1, 256, 0, s>>>(
                    static_cast<unsigned char const *>(a[0]), byte,
                    chunk_class_bytes(), wit);
            });
    }
    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        unsigned char *const host =
            static_cast<unsigned char *>(fx.hpool.addr(i));
        void *const dev = fx.gpu.addr(static_cast<ChunkId>(i));
        fx.plant_dev(dev, 0xDD); // an ungated copy carries the stale
        std::memset(host, 0xEE, chunk_class_bytes()); // plant, never the
                                                   // fill's bytes
    }

    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        unsigned char *const host =
            static_cast<unsigned char *>(fx.hpool.addr(i));
        void *const dev = fx.gpu.addr(static_cast<ChunkId>(i));
        TaskId const fill = kHostFillBase + static_cast<TaskId>(i);
        TaskId const check = kGpuCheckBase + static_cast<TaskId>(i);
        OpRef const tag = make_op_ref(0, i);

        fx.launch_host(fill, {host}); // the host producer's work
        EventHandle const f = fx.host.pool().acquire();
        fx.host.mint(f); // the host mint (the flag store)
        host_flags.push_back(f);

        fx.bridge.gate(kLaneChanH2G, tag, f, kLaneExecHost);
        fx.bridge.submit(kLaneChanH2G, tag, [&fx, host, dev] {
            fx.h2d.copy(host, dev, chunk_class_bytes()); // trails host progress
        });
        EventHandle const e = fx.pool.acquire();
        dev_handles.push_back(e);
        fx.bridge.submit(kLaneChanH2G, tag, [&fx, e] {
            fx.h2d.mint(e); // the record — enqueued strictly before
        });                  // the wait call below, in the lane order
        fx.bridge.submit(kLaneChanH2G, tag, [&fx, e, check, dev] {
            fx.gpu.wait(e); // the capture, at its call time
            std::vector<void *> a{dev};
            fx.gpu.launch(check, kGpu, a.data());
        });
    }
    fx.join_all(); // the device barriers drain the deferred tail first
    CHECK(fx.bridge.idle());

    for (std::size_t i = 0; i < kMintPairs; ++i)
    {
        CHECK(fx.read_wit(16 + i) == 1); // every deferred pair gated
    }
    for (EventHandle h : dev_handles)
    {
        fx.pool.release(h);
    }
    for (EventHandle h : host_flags)
    {
        fx.host.pool().release(h);
    }
}

// The across-devices leg, per direction on the probed pair: a mint on
// device A's exec lane waited by device B's lane, under the SAME
// deep-pipelining shape as the host-boundary phases — kPeerPairs
// pairs enqueued with no barrier, mints outstanding across two
// contexts, every byte pre-planted stale so a wait that failed to
// capture its record reads wrong bytes at the barrier.
//
// COMPOSITION CONSTRAINT (documented, a real seam this leg names):
// the wait surface resolves an EventHandle through the CALLED
// lane's own family pool (CudaLaneOps::wait -> its pool's raw()), so
// per-device pool composition  cannot
// express a cross-device wait — the consumer would resolve a foreign
// index in its own pool. On a 1-GPU host every CUDA lane shares
// device 0's pool and the seam is invisible; the 2-GPU body therefore
// provisions ONE shared pool per direction, created on A (every
// RECORD stays same-context, on A's exec stream) and every consumer —
// the channel's device-B stream AND B's exec lane — WAITS the shared
// process-scoped handle (cudaStreamWaitEvent across contexts, the one
// cross-context mechanism used here; a cross-context
// RECORD is not a supported mechanism and is never issued here). The
// channel mints nothing; the byte proof rides its gated copy.
constexpr std::size_t kPeerPairs = 24;

void conf_mint_capture_across_devices(std::uint8_t const *bits)
{
    for (int a = 0; a < 2; ++a)
    {
        int const b = 1 - a;
        std::size_t const bit = static_cast<std::size_t>(a) * 2u +
                                static_cast<std::size_t>(b);
        if (bits[bit] == 0u)
        {
            skip_pair_absent("conf_mint_capture across devices", a, b);
            continue;
        }

        // One shared family per direction (the constraint above): all
        // records land on A's exec stream, all waits are cross-context.
        CudaEventPool shared(a, 2 * kPeerPairs + 8);
        CudaExecLane lane_a(a, 0, kPeerPairs + 4, shared);
        CudaExecLane lane_b(b, 0, kPeerPairs + 4, shared);
        CudaPeerChannel chan(a, b, shared); // its stream is device B's

        // The per-pair fill symbols on A (a varying byte
        // multiplies task symbols, per node), and B's noop body for
        // the exec-lane wait's order-only work below.
        for (std::size_t i = 0; i < kPeerPairs; ++i)
        {
            unsigned char const byte = pat_byte(i + 129);
            lane_a.register_task(
                kGpuFillBase + static_cast<TaskId>(i),
                [byte](cudaStream_t s, void *const *args) {
                    cudaMemsetAsync(args[0], byte, chunk_class_bytes(), s);
                });
        }
        lane_b.register_task(kGpuNoop,
            [](cudaStream_t s, void *const *) {
                k_noop<<<1, 1, 0, s>>>();
            });

        // The stale plants on both devices (harness bytes, planted
        // before anything enqueues): an ungated wait copies A's 0xDD
        // or leaves B's 0xEE standing — either way the read-back at
        // the barrier names it.
        std::vector<void *> srcs;
        std::vector<void *> dsts;
        srcs.reserve(kPeerPairs);
        dsts.reserve(kPeerPairs);
        for (std::size_t i = 0; i < kPeerPairs; ++i)
        {
            srcs.push_back(lane_a.addr(static_cast<ChunkId>(i)));
            dsts.push_back(lane_b.addr(static_cast<ChunkId>(i)));
        }
        cudaSetDevice(a);
        for (void *p : srcs)
        {
            cudaMemset(p, 0xDD, chunk_class_bytes());
        }
        // plant_dev's comment: the plants join before the storm's
        // non-blocking work is issued.
        cudaStreamSynchronize(0);
        cudaSetDevice(b);
        for (void *p : dsts)
        {
            cudaMemset(p, 0xEE, chunk_class_bytes());
        }
        cudaStreamSynchronize(0);

        // The storm: fill on A, the mint right after the work, the
        // channel (device B's lane) waits the mint and carries the
        // bytes across, and B's exec lane waits the SAME mint before
        // its own work — pair after pair, no barrier, the records
        // strictly preceding their waits in the single issue order.
        std::vector<EventHandle> handles;
        handles.reserve(kPeerPairs);
        for (std::size_t i = 0; i < kPeerPairs; ++i)
        {
            TaskId const fill = kGpuFillBase + static_cast<TaskId>(i);

            std::vector<void *> s_args{srcs[i]};
            lane_a.launch(fill, a, s_args.data());
            EventHandle const e = shared.acquire();
            lane_a.mint(e); // the record, on A's stream
            handles.push_back(e);

            chan.wait(e); // B's stream gates on A's record
            chan.copy(srcs[i], dsts[i], chunk_class_bytes());

            lane_b.wait(e); // B's exec lane gates on the same mint —
                            // order-only work below; the gating's byte
                            // proof rides the channel's copy above
            std::vector<void *> b_args{dsts[i]};
            lane_b.launch(kGpuNoop, b, b_args.data());
        }
        lane_a.barrier();
        chan.barrier();
        lane_b.barrier();

        // The byte proof, read strictly at the barriers :
        // every destination holds its producer's bytes, whole class,
        // tail included — a wait that captured a not-yet-recorded
        // event would have let the stale plants through.
        std::vector<unsigned char> back(chunk_class_bytes(), 0);
        for (std::size_t i = 0; i < kPeerPairs; ++i)
        {
            if (cudaMemcpy(back.data(), dsts[i], chunk_class_bytes(),
                    cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                throw std::runtime_error(
                    "conf_mint_capture across devices: the read-back");
            }
            CHECK(host_chunk_eq(
                back.data(), pat_byte(i + 129)));
        }
        for (EventHandle h : handles)
        {
            shared.release(h);
        }
        cudaSetDevice(0);
    }
}

void conf_mint_capture()
{
    {
        ConfFx fx(kMintPairs + 8, 1024, 64);
        conf_mint_capture_phase_a(fx);
    }
    {
        ConfFx fx(kMintPairs + 8, 1024, 64);
        conf_mint_capture_phase_b(fx);
    }
    // The across-devices leg — a mint on device A's lane waited by
    // device B's lane — needs a second CUDA device ;
    // the test host holds one, so the leg prints its loud skip.
    // On a 2-GPU host it RUNS: the probed pair's lanes, the cross-
    // context record/wait storm, the byte checks above.
    if (!have_two_devices())
    {
        skip_needs_two_devices("conf_mint_capture across devices");
        return;
    }
    std::vector<std::uint8_t> const bits = prepare_two_device_peer();
    conf_mint_capture_across_devices(bits.data());
}

// ==========================================================================
// Case 8: conf_submission_latency — enqueue cost bounded under
// full lanes. 10k enqueues on the device lane (a deep queue pending:
// a ~15 ms spin kernel holds the head) and 10k on the host lane (the
// worker busy on a sleep task): amortized < 10 us/op — generous, the
// pin is non-blocking submission, not performance.
// ==========================================================================

void conf_submission_latency()
{
    constexpr std::size_t kEnqueues = 10000;
    constexpr double kBoundUsPerOp = 10.0;

    {
        ConfFx fx(4, 64, 4);
        // The pending head: a short spin kernel (~15 ms) keeps the
        // lane busy while the timed enqueues pile behind it — a deep
        // queue stands, without the driver's launch-channel backpressure
        // a long head would add to the timed window.
        std::vector<unsigned char> zeros(chunk_class_bytes(), 0);
        void *const chunk = fx.gpu.addr(0);
        if (cudaMemcpy(chunk, zeros.data(), chunk_class_bytes(),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("conf_submission_latency: plant");
        }
        std::vector<void *> a{chunk};
        fx.launch_gpu(kGpuSpinShort, a); // the deep pending head

        std::vector<EventHandle> handles;
        handles.reserve(kEnqueues / 10 + 1);
        for (std::size_t h = 0; h < 8; ++h)
        {
            handles.push_back(fx.pool.acquire());
        }
        auto const t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kEnqueues; ++i)
        {
            fx.launch_gpu(kGpuNoop, a);
            if (i % 10 == 0)
            {
                fx.gpu.mint(handles[(i / 10) % handles.size()]);
            }
        }
        auto const t1 = std::chrono::steady_clock::now();
        double const us_per_op =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count() /
            1000.0 / static_cast<double>(kEnqueues);
        std::cout << "[backend_conformance] conf_submission_latency: "
                  << "device lane " << us_per_op << " us/op over "
                  << kEnqueues << " enqueues (bound " << kBoundUsPerOp
                  << ")\n";
        CHECK(us_per_op < kBoundUsPerOp);
        fx.gpu.barrier();
        for (EventHandle h : handles)
        {
            fx.pool.release(h);
        }
    }

    {
        ConfFx fx(4, 64, 4);
        unsigned char *const cell =
            static_cast<unsigned char *>(fx.hpool.addr(0));
        std::memset(cell, 0, chunk_class_bytes());
        fx.launch_host(kHostSlow, {cell}); // the worker's pending head
        auto const t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kEnqueues; ++i)
        {
            fx.launch_host(kHostNoop, {cell});
        }
        auto const t1 = std::chrono::steady_clock::now();
        double const us_per_op =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count() /
            1000.0 / static_cast<double>(kEnqueues);
        std::cout << "[backend_conformance] conf_submission_latency: "
                  << "host lane " << us_per_op << " us/op over "
                  << kEnqueues << " enqueues (bound " << kBoundUsPerOp
                  << ")\n";
        CHECK(us_per_op < kBoundUsPerOp);
        fx.host.barrier();
    }
}

// ==========================================================================
// Case 9: conf_barrier_completeness — every notified op observed
// at the join, none lost. Each of the four lanes carries K ops, each
// op trailed by a notify stamping its own fixture slot; the barriers
// join, and the fires' count equals the ops exactly.
// ==========================================================================

void conf_barrier_completeness()
{
    constexpr std::size_t kPerLane = 48;
    constexpr std::size_t kLanes = 4;
    ConfFx fx(4, 64, 4);

    auto slots = std::make_unique<FireSlot[]>(kPerLane * kLanes);
    std::atomic<std::uint64_t> fires{0};
    // The fire payload rides the namespace-scope Fire2Arg/fire2_hook;
    // the deque keeps every argument's address stable while notifies
    // fly (the notification callback carries a bare pointer).
    std::deque<Fire2Arg> args;

    void *const dev0 = fx.gpu.addr(0);
    unsigned char *const host0 = static_cast<unsigned char *>(fx.hpool.addr(0));
    void *const dev1 = fx.gpu.addr(1);
    unsigned char *const host1 = static_cast<unsigned char *>(fx.hpool.addr(1));
    std::vector<void *> a{dev0};

    for (std::size_t i = 0; i < kPerLane; ++i)
    {
        // exec(0): a launch plus its fire.
        fx.launch_gpu(kGpuNoop, a);
        slots[i].expect = static_cast<std::uint32_t>(i + 1);
        args.push_back(Fire2Arg{&slots[i], &fires});
        fx.gpu.notify(&fire2_hook, &args.back());
        // exec(host): a launch plus its fire.
        fx.launch_host(kHostNoop, {host0});
        slots[kPerLane + i].expect =
            static_cast<std::uint32_t>(i + 1);
        args.push_back(Fire2Arg{&slots[kPerLane + i], &fires});
        fx.host.notify(&fire2_hook, &args.back());
        // chan(1,0): a copy plus its fire.
        fx.h2d.copy(host0, dev0, chunk_class_bytes());
        slots[2 * kPerLane + i].expect =
            static_cast<std::uint32_t>(i + 1);
        args.push_back(Fire2Arg{&slots[2 * kPerLane + i], &fires});
        fx.h2d.notify(&fire2_hook, &args.back());
        // chan(0,1): a copy plus its fire.
        fx.d2h.copy(dev1, host1, chunk_class_bytes());
        slots[3 * kPerLane + i].expect =
            static_cast<std::uint32_t>(i + 1);
        args.push_back(Fire2Arg{&slots[3 * kPerLane + i], &fires});
        fx.d2h.notify(&fire2_hook, &args.back());
    }
    fx.join_all();

    CHECK(fires.load() == kPerLane * kLanes); // the fire-ring count
    for (std::size_t i = 0; i < kPerLane * kLanes; ++i)
    {
        // Every op observed at the join, none lost, none doubled.
        CHECK(slots[i].stamp.load() == slots[i].expect);
    }
}

// ==========================================================================
// Case 10: conf_fault_propagation — an injected fault delivered at
// the barrier, no silent mint, dependents never run. The enqueue
// refusal (detection point 1) runs in-process; the async device fault
// (detection point 2, the device-fault release path through the
// deferral bridge) runs in a FORKED, alarm-guarded child so a
// regression fails the check instead of hanging ctest.
// ==========================================================================

void conf_fault_refusal()
{
    ConfFx fx(8, 64, 8);
    fx.build_executor(false);

    unsigned char *const canary =
        static_cast<unsigned char *>(fx.gpu.addr(2));
    fx.plant_dev(canary, 0x5C); // "never ran" reads as still-0x5C

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        plink(3, 0, kGpu, 1), // 0 (issues)
        plink(9, 0, kGpu, 2), // 1 (issues)
        ptile(kGpuFill77, kGpu, {popnd(3, Access::Write, kGpu, 1)}), // 2
        ptile(kUnregistered, kGpu,
            {popnd(9, Access::Write, kGpu, 2)}), // 3: refused at enqueue
        ptile(kGpuFill33, kGpu,
            {popnd(9, Access::Write, kGpu, 2)}), // 4: the never-run canary
        punlink(9, 0, kGpu, 2),                  // 5: never enqueued
        punlink(3, 0, kGpu, 1),                  // 6: never enqueued
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
    // The census names the op: the fault, the position, the task.
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    CHECK(caught.fault.find("position 3") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=3)") !=
          std::string::npos);
    CHECK(caught.fault.find("888") != std::string::npos);
    // The two LINKs and the healthy fill issued; nothing after.
    CHECK(caught.issue_prefix.size() == 3);
    CHECK(fx.ex->ops_issued() == 3);
    CHECK(fx.ex->plans_executed() == 0);
    // No silent mint: the plan held no cross-lane debt, so nothing was
    // minted at all — and the close never ran (nothing released).
    CHECK(fx.ex->mints_acquired_total() == 0);
    CHECK(fx.ex->handles_released_total() == 0);

    // Dependents never ran: the fixture drains the lanes itself and
    // the canary bytes stay 0x5C.
    fx.gpu.barrier();
    std::vector<unsigned char> back(chunk_class_bytes(), 1);
    fx.read_chunk(back, canary);
    CHECK(host_chunk_eq(back.data(), 0x5C));
}

// The child body: the async device fault, a mint behind it, a host
// consumer gated on that mint's flag, and a bridge gate parked behind
// it all — the close join delivers the fault's census at the barrier,
// the gated reads never run, no mint completed silently.
void conf_fault_device_child()
{
    ConfFx fx(8, 64, 8);
    fx.build_executor(false);

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        plink(2, 0, kGpu, 2),  // 0
        plink(2, 0, kHost, 2), // 1
        ptile(kGpuFaultBad, kGpu,
            {popnd(2, Access::Write, kGpu, 2)}), // 2: faults async —
                                               // the LAST op ever
                                               // enqueued on this
                                               // lane, so the barrier
                                               // census's best name
                                               // IS the faulting op
        pcopy(2, 0, {kGpu, 2}, {kHost, 2}), // 3
        ptile(kHostCheck33, kHost,
            {popnd(2, Access::Read, kHost, 2)}), // 4: gated, never runs
        plink(0, 0, kHost, 1), // 5
        plink(0, 0, kGpu, 3),  // 6
        ptile(kHostFill11, kHost,
            {popnd(0, Access::Write, kHost, 1)}), // 7
        pcopy(0, 0, {kHost, 1}, {kGpu, 3}), // 8: the bridge gate — a
                                           // device consumer of the
                                           // dead host mint's chain,
                                           // parked when the fault's
                                           // release must lift it
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
    // Delivered AT THE BARRIER (the detection point), the census
    // naming the faulting op and the parks' release.
    CHECK(caught.fault.find("issue-abort") != std::string::npos);
    CHECK(caught.fault.find("position 2") != std::string::npos);
    CHECK(caught.fault.find("OpRef(plan_seq=0, position=2)") !=
          std::string::npos);
    // The light close's probe: the FAULTING LANE's own terminal-event
    // join delivered the fault (exec(0) joins first -- the common
    // killer), before any trailing lane's drain could park on it.
    CHECK(caught.fault.find("exec(0)") != std::string::npos);
    CHECK(fx.ex->plans_executed() == 0);
    // No silent mint: the flag behind the faulted chain never fired,
    // so the host consumer's gate never opened — its witness stays 0
    // (a dependent that never ran, observed without touching the
    // poisoned context).
    CHECK(fx.host_wit[kWitHostCheck33] == 0);
}

void conf_fault_propagation()
{
    conf_fault_refusal();

    // The async device fault needs a fresh process (CUDA forbids a
    // fork after init): a fresh exec of this binary, alarm-guarded —
    // a regression hangs the child, SIGALRM kills it, and the check
    // FAILS instead of ctest timing out.
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
        alarm(60);
        execl(self, self, "--child-fault", static_cast<char *>(nullptr));
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

// ==========================================================================
// Case 11: conf_stamp_fidelity — the op's measured
// interval IS the tile it owns of its lane's issue order (the host
// lane's tiles are the worker's own clock reads — the slow host
// op's tile covers its known floor); the tiles tile each lane; the
// stamps gate nothing — the traced run's issue log is the queue,
// unchanged.
// ==========================================================================

void conf_stamp_fidelity()
{
    constexpr std::uint64_t kFloorNs = 20000000u; // 20 ms, well under
    // the ~40 ms of the slow host op (a sleep in its body).
    ConfFx fx(8, 64, 8);
    fx.build_executor(true); // the armed tracer, 4 lane-carrying ops

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        plink(9, 0, kGpu, 0),  // 0
        plink(9, 0, kHost, 1), // 1
        plink(8, 0, kGpu, 2),  // 2
        ptile(kGpuFill77, kGpu, {popnd(9, Access::Write, kGpu, 0)}), // 3
        pcopy(9, 0, {kGpu, 0}, {kHost, 1}),                          // 4
        ptile(kHostSlow, kHost,
            {popnd(9, Access::Read, kHost, 1)}), // 5: ~40 ms on the host
        ptile(kGpuSlow, kGpu,
            {popnd(8, Access::Write, kGpu, 2)}), // 6: ~40 ms on device
        punlink(9, 0, kHost, 1),                 // 7
        punlink(8, 0, kGpu, 2),                  // 8
        punlink(9, 0, kGpu, 0),                  // 9
    };
    fx.ex->execute(p);

    // The issue tiles: a measured record per lane-carrying op, the
    // tiles TILING each lane's issue order (a lane's consecutive
    // records share the boundary: one's end IS the next's start),
    // non-negative everywhere, and the records in the queue's order
    // (stamps never reordered anything). The host lane's tiles are
    // the worker's own clock reads at FIFO positions
    // — the deliberately slow host op's tile (issue -> the next FIFO
    // position's read) covers its ~40 ms of work because the worker
    // IS the lane position; a device lane's tile spans issue-to-next-
    // issue and carries no device-duration claim (the cost model in
    // src/cuda_backend.hh).
    std::vector<MeasuredTraceRecord> const &recs =
        fx.tracer->measured_records();
    CHECK(recs.size() == 4);
    CHECK(fx.tracer->fired_ops() == 4);
    for (std::size_t i = 0; i < recs.size(); ++i)
    {
        CHECK(recs[i].end_ns >= recs[i].start_ns);
        if (i > 0)
        {
            CHECK(recs[i - 1].plan_seq < recs[i].plan_seq ||
                recs[i - 1].op_position < recs[i].op_position);
        }
    }
    std::uint64_t host_dur = 0;
    for (MeasuredTraceRecord const &r : recs)
    {
        if (r.op_position == 5)
        {
            host_dur = r.end_ns - r.start_ns;
        }
    }
    CHECK(host_dur >= kFloorNs);
    // The lane tiling: same-lane consecutive records share the
    // boundary (one's end IS the next's start — the issue order).
    for (std::size_t i = 0; i + 1 < recs.size(); ++i)
    {
        if (recs[i].lane == recs[i + 1].lane)
        {
            // Same lane, consecutive in the issue order: the tile
            // boundary is SHARED.
            CHECK(recs[i].end_ns == recs[i + 1].start_ns);
        }
    }

    // Stamps gate nothing: the traced run issued exactly the queue,
    // byte results landed, and the reconciliation passed.
    CHECK(fx.ex->last_issue().issue_log.size() == p.queue.size());
    for (std::size_t i = 0; i < p.queue.size(); ++i)
    {
        CHECK(fx.ex->last_issue().issue_log[i] == p.queue[i]);
    }
    CHECK(fx.ex->last_issue().reconcile_ok);
    CHECK(fx.host_wit[kWitHostSlow] == 1); // the copied bytes reached
    std::vector<unsigned char> back(chunk_class_bytes(), 0);
    fx.read_chunk(back, fx.gpu.addr(2));
    unsigned magic = 0;
    std::memcpy(&magic, back.data(), sizeof(magic));
    CHECK(magic == 0xC0FFEEu); // the slow device op really ran
}

// ==========================================================================
// Case 12: conf_planned_issued — both halves over N lanes:
// the log half (issue log == queue, op for op) and the sync half
// (issued waits == the derived owed set), over the classic
// reduction's four lanes; then a full engine-driven batch through
// signal_graph_end/wait with the tracer armed, witnessed through
// test_hooks' ManagerStats.
// ==========================================================================

void conf_planned_issued_executor_half()
{
    ConfFx fx(8, 256, 8);
    fx.build_executor(false);

    Plan p;
    p.plan_seq = 0;
    p.queue = {
        plink(0, 0, kHost, 0), // 0: T0's host home
        plink(0, 0, kGpu, 1),  // 1: T0's gpu replica
        ptile(kHostFill11, kHost,
            {popnd(0, Access::Write, kHost, 0)}), // 2 (exec host)
        pcopy(0, 0, {kHost, 0}, {kGpu, 1}),       // 3 (chan 1,0: h2d)
        ptile(kGpuCheck11, kGpu,
            {popnd(0, Access::Read, kGpu, 1)}), // 4 (exec 0)
        plink(1, 0, kGpu, 2),                   // 5
        plink(1, 0, kHost, 3),                  // 6
        ptile(kGpuFill33, kGpu,
            {popnd(1, Access::Write, kGpu, 2)}), // 7 (exec 0)
        pcopy(1, 0, {kGpu, 2}, {kHost, 3}),      // 8 (chan 0,1: d2h)
        ptile(kHostCheck33, kHost,
            {popnd(1, Access::Read, kHost, 3)}), // 9 (exec host)
        punlink(0, 0, kGpu, 1),                  // 10
        punlink(0, 0, kHost, 0),                 // 11
        punlink(1, 0, kGpu, 2),                  // 12
        punlink(1, 0, kHost, 3),                 // 13
    };
    fx.ex->execute(p);

    // The log half: the issue log equals the queue op for op (value
    // equality), all four lanes touched.
    CHECK(fx.ex->last_issue().issue_log.size() == p.queue.size());
    for (std::size_t i = 0; i < p.queue.size(); ++i)
    {
        CHECK(fx.ex->last_issue().issue_log[i] == p.queue[i]);
    }
    CHECK(fx.ex->ops_issued() == 14);
    // The sync half: the issued waits equal the derived owed set —
    // one pure function run twice, compared for equality .
    CHECK(fx.ex->last_issue().issued_waits ==
          fx.ex->last_derivation().owed_set);
    CHECK(fx.ex->last_issue().reconcile_ok);
    // Hand-pinned rows over the N lanes: the h2d copy defers on the
    // host fill's mint, its device reader waits the copy's mint, the
    // d2h copy waits the device fill's mint, the host reader gates on
    // the copy's mint.
    CHECK(fx.ex->last_issue().issued_waits[3] ==
          std::vector<OpRef>{make_op_ref(0, 2)});
    CHECK(fx.ex->last_issue().issued_waits[4] ==
          std::vector<OpRef>{make_op_ref(0, 3)});
    CHECK(fx.ex->last_issue().issued_waits[8] ==
          std::vector<OpRef>{make_op_ref(0, 7)});
    CHECK(fx.ex->last_issue().issued_waits[9] ==
          std::vector<OpRef>{make_op_ref(0, 8)});
    // The bytes closed the loop through all four lanes.
    CHECK(fx.read_wit(kWitCheck11) == 1);
    CHECK(fx.host_wit[kWitHostCheck33] == 1);
}

// The engine-API codelet bodies (a tiny mixed batch, engine_smoke's
// shape): a device fill, a host scale, a device reader.
void cf_fill(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int *>(cl_args));
    auto *buf = static_cast<Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemsetAsync(buf->get_chunk_ptr<void>(i),
            pattern,
            buf->chunk_bytes(i),
            cuda_stream());
    }
}

void cf_scale(void **buffers, void *cl_args) noexcept
{
    (void)cl_args;
    auto *a = static_cast<Buffer *>(buffers[0]);
    auto *b = static_cast<Buffer *>(buffers[1]);
    unsigned int const *av = a->get_ptr<unsigned int>();
    unsigned int *bv = b->get_ptr<unsigned int>();
    std::size_t const n = b->nbytes() / sizeof(unsigned int);
    for (std::size_t i = 0; i < n; ++i)
    {
        bv[i] = av[i] + 1u;
    }
}

void cf_read(void **buffers, void *cl_args) noexcept
{
    auto *out = *static_cast<unsigned int **>(cl_args);
    auto *buf = static_cast<Buffer *>(buffers[0]);
    cudaMemcpyAsync(out,
        buf->get_ptr<unsigned int>(),
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        cuda_stream());
}

void conf_planned_issued_engine_half()
{
    // The traced engine batch: NNHAUL_TRACE arms the tracer at init,
    // the batch closes through signal_graph_end/wait, and the
    // ManagerStats witness carries planned == issued.
    char const *const trace_path = "/tmp/nnhaul-backend-conf-api.json";
    setenv("NNHAUL_TRACE", trace_path, 1);
    init(0, 1, std::size_t{32} << 20, 8 * chunk_class_bytes());

    Codelet fill("conf-fill", nullptr, cf_fill, nullptr);
    Codelet scale("conf-scale", cf_scale, nullptr, nullptr);
    Codelet reader("conf-read", nullptr, cf_read, nullptr);

    Handle a(1024);
    Handle b(1024);
    a.set_name("conf-a");
    b.set_name("conf-b");
    int const pattern = 0x3c;
    unsigned int seen = 0;
    unsigned int *seen_ptr = &seen;

    insert(fill,
        {{&a, Access::Write}},
        &pattern,
        sizeof(pattern),
        cuda_worker);
    insert(scale,
        {{&a, Access::Read}, {&b, Access::Write}},
        nullptr,
        0,
        host_worker());
    insert(reader,
        {{&b, Access::Read}},
        &seen_ptr,
        sizeof(seen_ptr));

    set_duration(fill, 0, 1000);
    set_duration(scale, 1, 2000);
    set_duration(reader, 0, 3000);
    set_channel_rate(0, 1, 1);
    set_channel_rate(1, 0, 1);

    CHECK(trace_enabled()); // the tracer is armed
    GraphEnd end = signal_graph_end();
    // ONE issue discipline for traced and untraced runs alike: the
    // hint issues the batch; wait() joins it (the stamps are read at
    // each plan's close, wherever the close happens).
    CHECK(end.issued_seq != 0);
    CHECK(end.run_pending);
    wait();

    CHECK(seen == 0x3c3c3c3du); // the bytes closed the loop
    ManagerStats m = manager_stats();
    CHECK(m.plan_walks == 1);
    CHECK(m.plan_ops_planned == m.plan_ops_issued);
    CHECK(m.plan_reconcile_ok);
    CHECK(m.tile_ops == 3);
    CHECK(m.copy_ops >= 2);
    CHECK(kernel_launches() == 2);

    shutdown();
    unsetenv("NNHAUL_TRACE");
    std::remove(trace_path);
}

void conf_planned_issued()
{
    conf_planned_issued_executor_half();
    conf_planned_issued_engine_half();
}

// ---- the runner ----

struct Case
{
    char const *name;
    void (*fn)();
};

} // namespace

int main(int argc, char **argv)
{
    if (argc > 1 && std::string(argv[1]) == "--child-fault")
    {
        conf_fault_device_child();
        std::cout << "backend_conformance: child device-fault case done ("
                  << g_checks_failed << " failed)\n";
        return g_checks_failed == 0 ? 0 : 1;
    }

    Case const cases[] = {
        {"conf_no_readback", conf_no_readback},
        {"conf_address_stability", conf_address_stability},
        {"conf_whole_chunk_copy", conf_whole_chunk_copy},
        {"conf_bounce_buffer", conf_bounce_buffer},
        {"conf_lane_order_rmw", conf_lane_order_rmw},
        {"conf_epoch_reuse", conf_epoch_reuse},
        {"conf_host_boundary_h2d", conf_host_boundary_h2d},
        {"conf_host_boundary_d2h", conf_host_boundary_d2h},
        {"conf_mint_capture", conf_mint_capture},
        {"conf_submission_latency", conf_submission_latency},
        {"conf_barrier_completeness", conf_barrier_completeness},
        {"conf_fault_propagation", conf_fault_propagation},
        {"conf_stamp_fidelity", conf_stamp_fidelity},
        {"conf_planned_issued", conf_planned_issued},
    };

    if (argc > 2 && std::string(argv[1]) == "--only")
    {
        for (Case const &c : cases)
        {
            if (std::string(c.name) == argv[2])
            {
                c.fn();
            }
        }
        std::cout << "only-case done (" << g_checks_failed
                  << " failed)\n";
        return g_checks_failed == 0 ? 0 : 1;
    }
    int failed_before = 0;
    for (Case const &c : cases)
    {
        failed_before = g_checks_failed;
        c.fn();
        if (g_checks_failed == failed_before)
        {
            std::cout << "[backend_conformance] " << c.name << " PASS\n";
        }
        else
        {
            std::cout << "[backend_conformance] " << c.name << " FAIL\n";
        }
    }

    if (g_checks_failed != 0)
    {
        std::cerr << "backend_conformance: " << g_checks_failed
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "backend_conformance: all cases passed\n";
    return 0;
}
