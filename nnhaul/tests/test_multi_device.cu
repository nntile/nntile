/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_multi_device.cu
 * The multi-device integration suite: one
 * function per named case, each skipping loudly — the printed notice,
 * a PASS — on a host with fewer CUDA devices than the case needs:
 *
 * 1. it_two_gpu_peer_copy — a 2-GPU peer COPY, byte-exact, tail bytes
 *    included;
 * 2. it_host_staged_chain — the same move host-staged when the pair
 *    is absent from the inventory: two hops, one residue UNLINK
 *    ;
 * 3. it_host_tile_op — a host-node TILE_OP reading a device chunk
 *    through a COPY , stamps from the worker's
 *    clock;
 * 4. it_cpu_gpu_mixed — CUDA and CPU codelets interleaved, the
 *    multi-implementation codelet dispatching per its TILE_OP nodes
 *    ;
 * 5. it_peer_absent_inventory — copy-no-channel censused at check
 *    time against an inventory whose probe said no, plus the staged
 *    alternative passing ;
 * 6. it_boundary_mismatch_nodes — a hand-broken entry state differing
 *    on ONE node's segment: boundary-mismatch before any op issues,
 *    every segment rendered .
 *
 * Cases 3-6 run on the classic 1-GPU+host host; cases 1-2 need two
 * devices and skip loudly where absent — with the skip's premise
 * itself verified (the device-count probe cross-checked against the
 * runtime's own inventory, the no-pair premise checked against the
 * channel matrix rule). Case 5 is pure check-mode data (nothing
 * device-specific runs in check mode) and runs anywhere.
 * Cases 1-4 share the suite's ONE engine lifecycle on any host (one
 * public init/shutdown over the probed device count — repeated cuDNN
 * handle create/destroy cycles fault intermittently inside this
 * stack's driver; the 2-GPU moves ride the same runtime, their
 * counters diffed per batch). GPU: the suite initializes CUDA — run
 * under the suite's GPU lock.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/cuda_backend.hh"
#include "../src/host_pool.hh"
#include "../src/lru_planner.hh"
#include "../src/plan_checker.hh"
#include "../src/plan_executor.hh"

#include "nnhaul/defs.h"
#include "nnhaul/nnhaul.hh"
#include "nnhaul/test_hooks.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---- the device-count probe  ----

// The one host introspection the suite needs. cudaGetDeviceCount
// enumerates without creating a context; every case that goes past
// its skip initializes CUDA through the public init.
int probe_devices()
{
    int n = 0;
    cudaError_t const err = cudaGetDeviceCount(&n);
    CHECK(err == cudaSuccess);
    if (err != cudaSuccess)
    {
        return 0;
    }
    return n;
}

// The loud skip, verbatim: the printed notice, then a PASS.
void skip_notice(char const *name, int needed, int have)
{
    std::cout << "[multi_device] SKIP " << name << ": needs " << needed
              << " CUDA devices, host has " << have << "\n";
}

// ---- small file helpers (the trace dump's reader) ----

std::string read_file(char const *path)
{
    std::FILE *f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        return std::string();
    }
    std::string data;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
    {
        data.append(buf, n);
    }
    std::fclose(f);
    return data;
}

// One X event per line, each carrying its cat, lane, lane_name, and
// kind — a line scan is enough (the dump's fixed shape).
bool trace_has_event(std::string const &doc, char const *cat,
    std::string const &lane_id, char const *kind)
{
    std::size_t pos = 0;
    while (pos < doc.size())
    {
        std::size_t const nl = doc.find('\n', pos);
        std::string const line = doc.substr(pos,
            nl == std::string::npos ? std::string::npos : nl - pos);
        pos = nl == std::string::npos ? doc.size() : nl + 1;
        if (line.find("\"ph\": \"X\"") == std::string::npos ||
            line.find(cat) == std::string::npos ||
            line.find("\"lane\": " + lane_id + ",") == std::string::npos ||
            line.find(kind) == std::string::npos)
        {
            continue;
        }
        return true;
    }
    return false;
}

// ---- the shared codelet bodies (the public API's kernel surface) ----

// The device fill: one byte pattern over the tile's chunks, each
// chunk's own payload .
void cuda_fill_body(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int const *>(cl_args));
    auto *buf = static_cast<Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemsetAsync(buf->get_chunk_ptr<void>(i),
            pattern,
            buf->chunk_bytes(i),
            cuda_stream());
    }
}

// The gather landing: the operand chunks (ON THE READING NODE — the
// binding the TILE_OP names) copied to a pinned host buffer on the
// codelet's own stream, so the test can scan every byte after wait().
struct GatherArgs
{
    unsigned char *dst;
    std::size_t stride; // the chunk class: chunk i lands at dst + i*stride
};

void cuda_gather_body(void **buffers, void *cl_args) noexcept
{
    auto const args = *static_cast<GatherArgs const *>(cl_args);
    auto *buf = static_cast<Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemcpyAsync(args.dst + i * args.stride,
            buf->get_chunk_ptr<void>(i),
            buf->chunk_bytes(i),
            cudaMemcpyDeviceToHost,
            cuda_stream());
    }
}

// The host scan: every payload byte against the pattern, the verdict
// into the test cell the cl_args blob carries.
struct HostCheckArgs
{
    int *result;
    unsigned char pattern;
};

void host_check_body(void **buffers, void *cl_args) noexcept
{
    auto const args = *static_cast<HostCheckArgs const *>(cl_args);
    auto *buf = static_cast<Buffer *>(buffers[0]);
    auto const *p = buf->get_ptr<unsigned char>();
    bool ok = true;
    for (std::size_t i = 0; i < buf->nbytes(); ++i)
    {
        ok = ok && p[i] == args.pattern;
    }
    *args.result = ok ? 1 : 0;
}

// The multi-implementation pair (case 4): the two bodies write
// DIFFERENT patterns, so the dispatched body per TILE_OP node reads
// straight off the bytes.
void cpu_pattern_body(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int const *>(cl_args));
    auto *buf = static_cast<Buffer *>(buffers[0]);
    std::memset(buf->get_ptr<void>(), pattern, buf->nbytes());
}

void cuda_pattern_body(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int const *>(cl_args));
    auto *buf = static_cast<Buffer *>(buffers[0]);
    cudaMemsetAsync(buf->get_ptr<void>(),
        pattern,
        buf->nbytes(),
        cuda_stream());
}

bool host_scan_is(unsigned char const *p, std::size_t nbytes,
    unsigned char pattern)
{
    for (std::size_t i = 0; i < nbytes; ++i)
    {
        if (p[i] != pattern)
        {
            return false;
        }
    }
    return true;
}

// ---- 1. it_two_gpu_peer_copy  ----

// The g0 -> g1 move over a TWO-chunk tile whose second chunk carries
// an intra-chunk tail (nbytes = class + class/4): one whole-class
// chunk plus the tail bytes, byte-exact at the far end. The classified
// copy log tells whether the move crossed the peer channel or staged
// through the host on this particular 2-GPU host (the probe decides);
// the peer form is asserted wherever the probe said yes.
struct MoveOutcome
{
    bool bytes_ok = false;
    // COPY-record counts of THIS move's batch: a COPY names ONE
    // logical chunk ,
    // so an n-chunk move is n COPYs per hop, n CopyRecords.
    int direct01 = 0; // COPY records g0 -> g1 (one per chunk)
    int staged02 = 0; // g0 -> host
    int staged21 = 0; // host -> g1
    std::uint64_t unlink_ops = 0; // the batch's UNLINKs (the residue)
    std::uint64_t tile_ops = 0;
};

// Runs INSIDE the suite's one engine lifecycle (main builds it): the
// per-move counters are DIFFS of the cumulative surfaces (copy_log,
// manager_stats) over the move's own batch, so two moves may ride one
// runtime — the suite never opens a second init/shutdown cycle on any
// host (the note-21 teardown fault's trigger).
MoveOutcome run_g0_g1_move()
{
    MoveOutcome out;
    std::size_t const tail = chunk_class_bytes() / 4;
    std::size_t const nbytes = chunk_class_bytes() + tail;
    CHECK(tile_chunk_count(nbytes) == 2); // the two-chunk move, pinned

    std::size_t const log_before = copy_log().size();
    ManagerStats const pre = manager_stats();

    unsigned char *landing = nullptr;
    if (cudaMallocHost(&landing, 2 * chunk_class_bytes()) != cudaSuccess)
    {
        throw std::runtime_error("it move: the pinned landing's alloc");
    }
    std::memset(landing, 0, 2 * chunk_class_bytes());

    Codelet fill("md-move-fill", nullptr, cuda_fill_body, nullptr);
    Codelet gather("md-move-gather", nullptr, cuda_gather_body, nullptr);

    Handle x(nbytes);
    x.set_name("md-move");
    int const pattern_int = 0x77;
    GatherArgs gather_args{landing, chunk_class_bytes()};

    // The producer on device node 0, the reader on device node 1 —
    // the cross-device dataflow arrives as the planner's COPYs
    // , the bytes verified at the far end.
    insert(fill,
        {{&x, Access::Write}},
        &pattern_int,
        sizeof(pattern_int),
        cuda_worker);
    insert(gather,
        {{&x, Access::Read}},
        &gather_args,
        sizeof(gather_args),
        1);
    signal_graph_end();
    wait();

    out.bytes_ok = host_scan_is(landing, nbytes, 0x77);
    cudaFreeHost(landing);

    std::vector<CopyRecord> const log = copy_log();
    // The chunk-span rule at the hook : the move's COPYs
    // carry the occupied span of their logical chunk — chunk 0 the
    // whole class, chunk 1 the quarter-class tail — and nothing else,
    // each span present among the batch's records (both chunks move).
    std::size_t const tail_span = chunk_class_bytes() / 4;
    bool saw_class_span = false;
    bool saw_tail_span = false;
    for (std::size_t i = log_before; i < log.size(); ++i)
    {
        CopyRecord const &rec = log[i];
        CHECK(rec.nbytes == chunk_class_bytes() ||
              rec.nbytes == tail_span);
        if (rec.nbytes == chunk_class_bytes())
        {
            saw_class_span = true;
        }
        if (rec.nbytes == tail_span)
        {
            saw_tail_span = true;
        }
        if (rec.src_node == 0 && rec.dst_node == 1)
        {
            out.direct01 += 1;
        }
        if (rec.src_node == 0 && rec.dst_node == 2)
        {
            out.staged02 += 1;
        }
        if (rec.src_node == 2 && rec.dst_node == 1)
        {
            out.staged21 += 1;
        }
    }
    CHECK(saw_class_span);
    CHECK(saw_tail_span);
    ManagerStats const post = manager_stats();
    out.unlink_ops = post.unlink_ops - pre.unlink_ops;
    out.tile_ops = post.tile_ops - pre.tile_ops;

    x.invalidate();
    wait();
    return out;
}

void it_two_gpu_peer_copy(int devices)
{
    if (devices < 2)
    {
        // The skip's premise VERIFIED, not just printed: the probe is
        // deterministic — a second cudaGetDeviceCount agrees — and the
        // count is the runtime's own node count, cross-checked where
        // the suite's ONE engine lifecycle runs (it_host_tile_op's
        // init over exactly the probed count: k device nodes plus the
        // host node, the host node's id IS k). Keeping the
        // suite to one public init/shutdown cycle sidesteps an
        // intermittent driver-side fault this stack shows inside
        // cudnnDestroy on the third handle create/destroy cycle in
        // one process (observed here; libcudnn_graph.so's teardown
        // racing libcuda — never the engine's own path).
        int const again = probe_devices();
        CHECK(again == devices);
        CHECK(devices >= 0);
        skip_notice("it_two_gpu_peer_copy", 2, devices);
        return;
    }

    // A 2-GPU host: the probe of the ordered pair decides the form
    // (the runtime's own probe at init is the same query, recorded
    // faithfully, never mirrored).
    int can_peer = 0;
    cudaError_t const err = cudaDeviceCanAccessPeer(&can_peer, 0, 1);
    CHECK(err == cudaSuccess);

    MoveOutcome const out = run_g0_g1_move();

    // Byte-exact, tail bytes included: both chunks' every payload
    // byte verified at the far end, the second chunk's payload being
    // the intra-chunk tail.
    CHECK(out.bytes_ok);
    CHECK(out.tile_ops == 2); // the fill and the gather, one node each

    if (can_peer != 0)
    {
        // The peer COPY: one direct g0 -> g1 COPY PER LOGICAL CHUNK
        // (a COPY names one chunk; the two-chunk move is two COPYs),
        // nothing staged.
        CHECK(out.direct01 == 2);
        CHECK(out.staged02 == 0 && out.staged21 == 0);
    }
    else
    {
        // The pair absent: the staged alternative is the move that
        // runs  — two hops PER LOGICAL CHUNK (fragment
        // 2's form is per chunk: COPY g0->host + COPY host->g1, one
        // residue UNLINK), so four COPY records and two residue
        // UNLINKs; it_two_gpu_peer_copy's direct form is unreachable
        // on this host, and case 5 pins the copy-no-channel rule
        // behind it.
        CHECK(out.staged02 == 2 && out.staged21 == 2);
        CHECK(out.direct01 == 0);
        CHECK(out.unlink_ops == 2); // the staging residue, per chunk
    }
}

// ---- 2. it_host_staged_chain  ----

void it_host_staged_chain(int devices)
{
    if (devices < 2)
    {
        // The premise VERIFIED, data-only: on one device node plus
        // the host there IS no device pair — the channel matrix's
        // only pairs are device<->host both directions ,
        // so the g0 -> g1 two-hop chain has no referent to stage.
        // The staged FORM itself — two hops, one residue UNLINK — is
        // pinned on this host by it_peer_absent_inventory (check
        // mode, data only) and by it_host_tile_op's real one-hop
        // staging through the host.
        NodeInventory const one(1,
            std::vector<std::size_t>{2, 2},
            std::vector<std::uint8_t>{0});
        CHECK(one.node_count() == 2);
        CHECK(one.is_device(0) && one.is_host(1));
        CHECK(one.has_channel(0, 1) && one.has_channel(1, 0));
        CHECK(!one.node_valid(2)); // no second device node exists
        skip_notice("it_host_staged_chain", 2, devices);
        return;
    }

    // A 2-GPU host: the pair the probe reports decides. Absent — the
    // case's own subject — the same g0 -> g1 move IS the host-staged
    // chain: COPY g0 -> host, COPY host -> g1, and the staging
    // residue's UNLINK .
    int can_peer = 0;
    cudaError_t const err = cudaDeviceCanAccessPeer(&can_peer, 0, 1);
    CHECK(err == cudaSuccess);

    MoveOutcome const out = run_g0_g1_move();
    CHECK(out.bytes_ok); // the staged move lands byte-exact too

    if (can_peer == 0)
    {
        // The host-staged form, PER LOGICAL CHUNK (a COPY names one
        // chunk): two hops and one residue UNLINK per chunk —
        // four COPY records, two residue UNLINKs over the two-chunk
        // move.
        CHECK(out.staged02 == 2 && out.staged21 == 2);
        CHECK(out.direct01 == 0);
        CHECK(out.unlink_ops == 2);
    }
    else
    {
        // The pair present: the runtime takes the one-hop peer price
        //  — the staged chain is then
        // this host's impossible shape, pinned against a no-pair
        // inventory by it_peer_absent_inventory below.
        CHECK(out.direct01 == 2);
        std::cout << "[multi_device] NOTE it_host_staged_chain: peer "
                     "pair present, move served by the direct channel; "
                     "the staged form is pinned by "
                     "it_peer_absent_inventory\n";
    }
}

// ---- 3. it_host_tile_op  ----

// The 1-GPU+host realization of fragment 3, through the PUBLIC API:
// a fill TILE_OP on device node 0, a CPU reader TILE_OP on the host
// node (explicit host_worker()), the planner's COPY g0 -> host
// between them — the host TILE_OP reads a device chunk through a
// COPY, and its stamps come from the host worker's own clock
// , verified in the traced dump. Runs inside the
// suite's ONE engine lifecycle (main builds it over the probed
// device count with NNHAUL_TRACE armed; on a 2-GPU host the moves of
// cases 1-2 rode the same runtime first, so this case's counters are
// DIFFS over its own batch).
void it_host_tile_op(int devices)
{
    // The skip logic's runtime cross-check (the premise it_two_gpu_
    // peer_copy's skip names): init over the probed count builds
    // exactly k device nodes plus the host node, the host node's id
    // IS k  — the probe's count and the runtime's agree.
    CHECK(trace_enabled());
    CHECK(stats().memory_nodes == devices + 1);
    CHECK(host_worker() == devices);

    // The lane arithmetic, from the same inventory rule the engine
    // builds: exec lanes 0..k, then the channel lanes ascending over
    // the connected ordered pairs. The data-side twin reproduces the
    // runtime's matrix (the same probe the engine made); the test
    // host's pinned ids —
    // exec(host) = 1, chan(0,host) = 2 — asserted where k = 1.
    std::vector<std::uint8_t> peer(
        static_cast<std::size_t>(devices) * static_cast<std::size_t>(devices),
        0);
    for (int i = 0; i < devices; ++i)
    {
        for (int j = 0; j < devices; ++j)
        {
            if (i == j)
            {
                continue;
            }
            int can = 0;
            CHECK(cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess);
            peer[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(devices) +
                static_cast<std::size_t>(j)] = can != 0 ? 1 : 0;
        }
    }
    NodeInventory const probe(static_cast<std::size_t>(devices),
        std::vector<std::size_t>(
            static_cast<std::size_t>(devices) + 1, 1),
        peer);
    LaneId const host_lane = probe.lane_of_exec(probe.host_id());
    LaneId const d2h_lane = probe.lane_of_chan(0, probe.host_id());
    if (devices == 1)
    {
        CHECK(host_lane == 1 && d2h_lane == 2);
    }

    ManagerStats const pre = manager_stats();
    std::size_t const log_before = copy_log().size();

    Codelet fill("md-ht-fill", nullptr, cuda_fill_body, nullptr);
    Codelet reader("md-ht-read", host_check_body, nullptr, nullptr);

    Handle a(std::size_t{1} << 20); // one single-chunk tile
    a.set_name("md-ht");
    int const pattern_int = 0x5A;
    int read_ok = -1;
    HostCheckArgs check_args{&read_ok, 0x5A};

    insert(fill,
        {{&a, Access::Write}},
        &pattern_int,
        sizeof(pattern_int),
        cuda_worker);
    insert(reader,
        {{&a, Access::Read}},
        &check_args,
        sizeof(check_args),
        host_worker());
    GraphEnd const end = signal_graph_end();
    CHECK(end.issued_seq != 0); // the hint ISSUES the batch — one
    // discipline for traced and untraced runs alike (the stamps are
    // read at each plan's close, wherever the close happens).
    CHECK(end.run_pending);
    wait();

    // The host reader saw the device writer's bytes through the COPY
    // and its derived wait — fragment 3's TILE_OP-on-host reading a
    // device chunk.
    CHECK(read_ok == 1);

    // The plan's shape, over THIS case's batch: two TILE_OPs and at
    // least the one g0 -> host COPY between them; every logged
    // transfer of the batch crossed a pair and carried the copied
    // chunk's OCCUPIED SPAN  — this tile is one
    // single chunk of 1 MiB, a tail chunk: its span is its payload,
    // not the class.
    ManagerStats const m = manager_stats();
    CHECK(m.plan_walks == pre.plan_walks + 1);
    CHECK(m.plan_reconcile_ok);
    CHECK(m.tile_ops == pre.tile_ops + 2);
    CHECK(m.copy_ops >= pre.copy_ops + 1);
    bool saw_d2h = false;
    std::vector<CopyRecord> const log = copy_log();
    for (std::size_t i = log_before; i < log.size(); ++i)
    {
        CopyRecord const &rec = log[i];
        CHECK(rec.src_node != rec.dst_node);
        CHECK(rec.nbytes == tile_chunk_span(std::size_t{1} << 20, 0));
        if (rec.src_node == 0 && rec.dst_node == host_worker())
        {
            saw_d2h = true;
        }
    }
    CHECK(saw_d2h);

    // The stamps, from the worker's clock: the traced dump carries a
    // measured record for the host lane's TILE_OP (the worker's own
    // before/after clock reads), the d2h COPY on its channel lane, and
    // the device fill on exec(0); the full frame
    // names the host lane.
    char const *const mid_path = "/tmp/nnhaul-md-host-tile-op-mid.json";
    CHECK(trace_dump(mid_path));
    std::string const doc = read_file(mid_path);
    CHECK(!doc.empty());
    CHECK(doc.find(probe.lane_name(host_lane)) != std::string::npos);
    CHECK(trace_has_event(doc, "\"cat\": \"exec\"",
        std::to_string(host_lane), "\"kind\": \"tile_op\""));
    CHECK(trace_has_event(doc, "\"cat\": \"chan\"",
        std::to_string(d2h_lane), "\"kind\": \"copy\""));
    CHECK(trace_has_event(doc, "\"cat\": \"exec\"", "0",
        "\"kind\": \"tile_op\""));

    // The bytes once more through the public host acquire.
    {
        auto pinned = a.acquire(Access::Read);
        CHECK(host_scan_is(static_cast<unsigned char const *>(pinned.get()),
            std::size_t{1} << 20,
            0x5A));
    }

    a.invalidate();
    wait();
}

// ---- 4. it_cpu_gpu_mixed  ----

// ONE codelet, BOTH bodies (the multi-implementation codelet),
// inserted twice — once with the default worker resolution (the
// lowest-numbered runnable node: device node 0) and once with
// an explicit host_worker(). The TILE_OP's node selects the body
// : the bytes each body writes differently prove which
// body ran on which node. Rides the same runtime as case 3 (traced —
// the hint defers the close to wait()).
void it_cpu_gpu_mixed()
{
    ManagerStats const pre = manager_stats();
    std::uint64_t const pre_launches = kernel_launches();

    Codelet both("md-mixed", cpu_pattern_body, cuda_pattern_body, nullptr);

    Handle d(std::size_t{1} << 20); // the default insert's tile
    Handle h(std::size_t{1} << 20); // the explicit host insert's tile
    d.set_name("md-mixed-d");
    h.set_name("md-mixed-h");
    int const cuda_pattern = 0x22;
    int const cpu_pattern = 0x44;

    // Interleaved in one batch: CUDA then CPU, submit order carried
    // verbatim into the plan's queue .
    insert(both,
        {{&d, Access::Write}},
        &cuda_pattern,
        sizeof(cuda_pattern)); // default -> device node 0
    insert(both,
        {{&h, Access::Write}},
        &cpu_pattern,
        sizeof(cpu_pattern),
        host_worker()); // explicit -> the host node
    GraphEnd const end = signal_graph_end();
    CHECK(end.issued_seq != 0); // the hint ISSUES the batch — one
    // discipline for traced and untraced runs alike.
    CHECK(end.run_pending);
    wait();

    // Only the device TILE_OP is a kernel launch; the CPU body runs
    // on the host lane and counts nothing (the per-node counter).
    CHECK(kernel_launches() == pre_launches + 1);

    ManagerStats const m = manager_stats();
    CHECK(m.plan_walks == pre.plan_walks + 1);
    CHECK(m.plan_reconcile_ok);
    CHECK(m.tile_ops == pre.tile_ops + 2);

    // The dispatch verdict, off the bytes: the default insert ran the
    // CUDA body (0x22 on the device node), the explicit insert the
    // CPU body (0x44 on the host node) — crossed dispatch would swap
    // them.
    {
        auto pd = d.acquire(Access::Read);
        CHECK(host_scan_is(static_cast<unsigned char const *>(pd.get()),
            std::size_t{1} << 20,
            0x22));
    }
    {
        auto ph = h.acquire(Access::Read);
        CHECK(host_scan_is(static_cast<unsigned char const *>(ph.get()),
            std::size_t{1} << 20,
            0x44));
    }

    // The task views carry the resolved nodes and their classes
    // .
    int device_views = 0;
    int host_views = 0;
    for (TaskView const &v : tasks())
    {
        if (v.codelet != "md-mixed" || v.phase != TaskPhase::Done)
        {
            continue;
        }
        if (v.worker == 0 && v.kind == Kind::Cuda)
        {
            device_views += 1;
        }
        if (v.worker == host_worker() && v.kind == Kind::Cpu)
        {
            host_views += 1;
        }
    }
    CHECK(device_views == 1);
    CHECK(host_views == 1);

    d.invalidate();
    h.invalidate();
    wait();
}

// ---- 5. it_peer_absent_inventory ----

// Check mode is data only : the inventory constructor
// takes the peer matrix as data, so a two-device inventory whose
// "probe" said no — fabricated, touching no device — feeds the
// CHECKER directly. Half A: a plan naming the absent pair's direct
// COPY censuses copy-no-channel at check time, the census rendering
// every node's segment. Half B: the planner's staged alternative over
// the same inventory — two hops through the host, one residue UNLINK
// — passes the same check.
void it_peer_absent_inventory()
{
    constexpr NodeId kG0 = 0;
    constexpr NodeId kG1 = 1;
    constexpr NodeId kHost = 2;
    constexpr TileId kX = 0;
    constexpr TaskId kReader = 3;

    NodeInventory const inv(2,
        std::vector<std::size_t>{2, 2, 4},
        std::vector<std::uint8_t>{0, 0, 0, 0}); // the probe said no
    CHECK(inv.node_count() == 3);
    CHECK(inv.host_id() == kHost);
    CHECK(inv.has_channel(kG0, kHost) && inv.has_channel(kHost, kG0));
    CHECK(inv.has_channel(kG1, kHost) && inv.has_channel(kHost, kG1));
    CHECK(!inv.has_channel(kG0, kG1) && !inv.has_channel(kG1, kG0));

    TileTable tiles;
    tiles.add(kX, chunk_class_bytes()); // n = 1
    ExecutabilityTable runnable;
    runnable.add(kReader, std::vector<NodeId>{kG1}); // task 3 runs on g1

    auto eb = [](NodeId node, ChunkId chunk, TileId tile, bool actual) {
        EntryBinding b;
        b.node = node;
        b.chunk = chunk;
        b.tile = tile;
        b.index = 0;
        b.actual = actual;
        return b;
    };

    // ---- half A: the direct COPY censused at check time ----

    Plan direct;
    direct.plan_seq = 0;
    direct.entry = {eb(kG0, 0, kX, true), eb(kG1, 0, kX, false)};
    direct.queue = {make_copy(kX, 0, ChunkLoc{kG0, 0}, ChunkLoc{kG1, 0})};

    std::vector<SeqTask> const no_tasks;
    CheckInput in;
    in.plan = &direct;
    in.inventory = &inv;
    in.tiles = &tiles;
    in.runnable = &runnable;
    in.task_sequence = &no_tasks;
    CheckResult const r = check(in);

    CHECK(!r.pass);
    CHECK(r.fault == "copy-no-channel"); // the fault's name verbatim
    CHECK(r.site == CheckSite::Queue);
    CHECK(r.position == 0);
    CHECK(r.op.has_value());
    CHECK(r.op->kind == PlanOpKind::Copy);
    CHECK(r.census.has_value());
    // The census renders EVERY node's segment : three
    // segments — g0, g1, host — each with its pool and free list.
    CHECK(r.census->fold.size() == 3);
    for (std::size_t n = 0; n < r.census->fold.size(); ++n)
    {
        CHECK(r.census->fold[n].node == n);
        CHECK(r.census->fold[n].pool == inv.pool_cardinality(n));
    }
    CHECK(r.census->fold[0].bindings.size() == 1); // X, actual on g0
    CHECK(r.census->fold[0].bindings[0].tile == kX);
    CHECK(r.census->fold[0].bindings[0].actual);
    CHECK(r.census->fold[1].bindings.size() == 1); // X, outdated on g1
    CHECK(!r.census->fold[1].bindings[0].actual);
    CHECK(r.census->fold[2].bindings.empty()); // the host holds nothing
    CHECK(r.census->fold[0].free_list.size() == 1);
    CHECK(r.census->fold[1].free_list.size() == 1);
    CHECK(r.census->fold[2].free_list.size() == 4);
    // Nothing folded before op 0; a failed check ships no exit state.
    CHECK(r.census->prefix.empty());
    CHECK(r.exit.empty());
    CHECK(r.census->entry == direct.entry);

    // ---- half B: the staged alternative passes the check ----

    PlanRequest req;
    req.plan_seq = 1;
    req.entry = {eb(kG0, 0, kX, true)}; // X actual on g0
    PlannedTask read_on_g1;
    read_on_g1.task = kReader;
    read_on_g1.node = kG1; // the read happens on g1: the move to stage
    read_on_g1.operands = {TileUse{kX, Access::Read}};
    req.tasks = {read_on_g1};

    LruPlanner planner;
    PlannedBatch const staged = planner.plan(req, inv, tiles);

    // The staged shape : the two hops
    // through the host, the residue UNLINK, and NO direct pair COPY.
    int g0_host = 0;
    int host_g1 = 0;
    bool residue_unlink = false;
    bool direct_copy = false;
    bool reader_op = false;
    for (PlanOp const &op : staged.plan.queue)
    {
        if (op.kind == PlanOpKind::Copy)
        {
            if (op.u.copy.src.node == kG0 && op.u.copy.dst.node == kHost)
            {
                g0_host += 1;
            }
            if (op.u.copy.src.node == kHost && op.u.copy.dst.node == kG1)
            {
                host_g1 += 1;
            }
            if (op.u.copy.src.node == kG0 && op.u.copy.dst.node == kG1)
            {
                direct_copy = true;
            }
        }
        if (op.kind == PlanOpKind::Unlink &&
            op.u.unlink.node == kHost)
        {
            residue_unlink = true;
        }
        if (op.kind == PlanOpKind::TileOp && op.u.tile_op.task == kReader &&
            op.u.tile_op.node == kG1)
        {
            reader_op = true;
        }
    }
    CHECK(g0_host == 1);
    CHECK(host_g1 == 1);
    CHECK(!direct_copy);
    CHECK(residue_unlink);
    CHECK(reader_op);

    // The same checker, the real executability row: PASS, and the
    // exit state is the two actual replicas (the residue gone).
    std::vector<SeqTask> const seq = to_seq_tasks(req.tasks);
    CheckInput in2;
    in2.plan = &staged.plan;
    in2.inventory = &inv;
    in2.tiles = &tiles;
    in2.runnable = &runnable;
    in2.task_sequence = &seq;
    CheckResult const r2 = check(in2);
    CHECK(r2.pass);
    CHECK(r2.fault.empty());
    CHECK(!r2.census.has_value());
    CHECK((r2.exit ==
        std::vector<EntryBinding>{eb(kG0, 0, kX, true),
            eb(kG1, 0, kX, true)}));
}

// ---- 6. it_boundary_mismatch_nodes ----

// The classic 1-GPU+host fixture (the plan_executor suite's own
// shape): real lanes, one executor, a planted applied-table
// difference on ONE node's segment — the HOST node's, the mirror of
// the executor suite's device-node plant — and the hand-broken entry
// state that differs exactly there.
namespace boundary
{

constexpr NodeId kGpu = 0;
constexpr NodeId kHost = 1;
constexpr std::size_t kChunksPerNode = 4;
constexpr std::size_t kProvision = 64;

struct Fx
{
    NodeInventory inv;
    CudaEventPool pool;
    CudaExecLane gpu;
    HostWorkerLane host;
    DeferralBridge bridge; // after the host lane, before the CUDA lanes
    HostPool hpool;
    CudaHostChannel d2h; // chan(0,1): gpu -> host
    CudaHostChannel h2d; // chan(1,0): host -> gpu
    // The tile table  the executor and tracer borrow: this
    // fixture's plan carries no COPY, so the table stands empty — the
    // chunk-span rule  is the plan_executor suite's to pin.
    TileTable tiles;
    std::unique_ptr<PlanTrace> tracer;
    std::unique_ptr<PlanExecutor> ex;
    cudaStream_t mint_stream = nullptr; // the twins' record stream

    Fx() :
        inv(1, {kChunksPerNode, kChunksPerNode}, {0}),
        pool(0, kProvision),
        gpu(0, 0, kChunksPerNode, pool),
        host(kProvision,
            [](TaskId, void *const *) {
                throw std::runtime_error(
                    "boundary fixture: no host task is ever issued");
            }),
        bridge(host),
        hpool(kChunksPerNode),
        d2h(0, HostChannelDirection::DeviceToHost, pool),
        h2d(0, HostChannelDirection::HostToDevice, pool)
    {
        tracer = std::make_unique<PlanTrace>(inv, tiles);
        tracer->arm("/tmp/nnhaul-md-boundary-trace.json");
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

    ~Fx()
    {
        // The abort leaves the lanes un-drained; join them here
        // before the members unwind.
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
        std::remove("/tmp/nnhaul-md-boundary-trace.json");
    }

    Fx(Fx const &) = delete;
    Fx &operator=(Fx const &) = delete;
};

} // namespace boundary

void it_boundary_mismatch_nodes(int devices)
{
    using boundary::Fx;
    using boundary::kGpu;
    using boundary::kHost;

    if (devices < 1)
    {
        skip_notice("it_boundary_mismatch_nodes", 1, devices);
        return;
    }

    Fx fx;

    // The planted difference on the HOST node's segment (the run's
    // initial load through the applied table's own transitions): tile 7
    // linked and ACTUAL on the host chunk.
    fx.ex->applied().link(kHost, 0, LogicalChunk{7, 0});
    fx.ex->applied().actualize(kHost, 0);

    // The hand-broken entry state: the same binding with the FLAG
    // FLIPPED — one bit, one node's segment; the gpu's segment agrees
    // (both free).
    Plan p;
    p.plan_seq = 0;
    EntryBinding b;
    b.node = kHost;
    b.chunk = 0;
    b.tile = 7;
    b.index = 0;
    b.actual = false; // the applied table says actual — the flip
    p.entry = {b};
    // A queue of binding-table-only ops (nothing launches; the abort
    // precedes op 0 whatever the queue carries).
    p.queue = {make_link(8, 0, kGpu, 1), make_unlink(8, 0, kGpu, 1)};

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
    CHECK(caught.fault.find("boundary-mismatch") != std::string::npos);

    // BEFORE any op issues: no op, no plan, no stamp, no fire, no
    // mint .
    CHECK(fx.ex->ops_issued() == 0);
    CHECK(fx.ex->plans_executed() == 0);
    CHECK(fx.ex->mints_acquired_total() == 0);
    CHECK(fx.tracer->measured_records().empty());
    CHECK(fx.tracer->fired_ops() == 0);

    // EVERY segment rendered, in BOTH census columns: the gpu's segment
    // free, the host's carrying the planted
    // binding, each with its pool and free list beside it.
    CHECK(caught.fold.size() == 2);
    CHECK(caught.applied.size() == 2);
    CHECK(caught.fold[0].node == kGpu);
    CHECK(caught.fold[0].pool == 4);
    CHECK(caught.fold[0].bindings.empty());
    CHECK(caught.fold[0].free_list.size() == 4);
    CHECK(caught.fold[1].node == kHost);
    CHECK(caught.fold[1].pool == 4);
    CHECK(caught.fold[1].bindings.size() == 1);
    CHECK(caught.fold[1].bindings[0].tile == 7);
    CHECK(caught.fold[1].bindings[0].actual); // the planted flag
    CHECK(caught.fold[1].free_list.size() == 3);

    // The census travels with the entry and the full queue prefix (a
    // plan-level fault: nothing applied).
    CHECK(caught.entry == p.entry);
    CHECK(caught.prefix == p.queue);
    CHECK(caught.issue_prefix.empty());
}

} // namespace

int main()
{
    int const devices = probe_devices();
    std::cout << "[multi_device] host has " << devices
              << " CUDA device(s)\n";

    // Cases 1-4 share the suite's ONE engine lifecycle on any host —
    // init once over the probed device count, shutdown once (an
    // intermittent driver-side fault inside cudnnDestroy shows on
    // this stack from the third handle create/destroy cycle in one
    // process; one cycle keeps the suite clear of it everywhere). On
    // a 2-GPU host the moves of cases 1-2 ride the same runtime, each
    // move's counters DIFFED over its own batch; with fewer devices
    // they skip loudly inside. The tracer stays armed throughout:
    // case 3 verifies the host-lane stamps in the dump.
    if (devices >= 1)
    {
        char const *const trace_path = "/tmp/nnhaul-md-host-tile-op.json";
        std::remove(trace_path);
        std::remove("/tmp/nnhaul-md-host-tile-op-mid.json");
        setenv("NNHAUL_TRACE", trace_path, 1); // read at init
        init(0, devices, std::size_t{32} << 20, 4 * chunk_class_bytes());
        it_two_gpu_peer_copy(devices);
        it_host_staged_chain(devices);
        it_host_tile_op(devices);
        it_cpu_gpu_mixed();
        shutdown();
        // The armed path's shutdown dump landed too; leave nothing
        // behind.
        {
            std::FILE *const f = std::fopen(trace_path, "rb");
            CHECK(f != nullptr);
            if (f != nullptr)
            {
                std::fclose(f);
            }
        }
        std::remove(trace_path);
        std::remove("/tmp/nnhaul-md-host-tile-op-mid.json");
        unsetenv("NNHAUL_TRACE");
    }
    else
    {
        skip_notice("it_two_gpu_peer_copy", 2, devices);
        skip_notice("it_host_staged_chain", 2, devices);
        skip_notice("it_host_tile_op", 1, devices);
        skip_notice("it_cpu_gpu_mixed", 1, devices);
    }

    it_peer_absent_inventory();
    it_boundary_mismatch_nodes(devices);

    if (g_checks_failed != 0)
    {
        std::cerr << "multi_device: " << g_checks_failed
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "multi_device: all checks passed\n";
    return 0;
}
