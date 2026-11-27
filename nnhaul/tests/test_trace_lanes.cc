/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_trace_lanes.cc
 * Interval capture over the inventory-derived lane set under real pool
 * pressure: a working set larger than the device cap forces staged
 * hops and evictions, so the dumped trace must show the exec lanes AND
 * both channel lanes with the planned == issued witness, plus a stats
 * sidecar (the manager and lane counters from the test_hooks APIs)
 * that the pipeline test cross-checks against the dump. The old
 * three-stream constants and the C memory counters died with the
 * reactive engine (their workload is test_trace_mem's, deleted).
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

// Deterministic pressured workload over the four lanes of one device
// plus host. Each tile is host-written first (its version 1 lives in a
// host block — nothing on the device); the chain then READS it on the
// device (the planner stages a COPY host->device on chan(1,0)), WRITES
// it there (a device-only version 2 — the 12-tile working set against
// the 4-block cap forces evictions of earlier tiles), and reads the NEW
// version back with a HOST task (the planner must COPY device->host on
// chan(0,1) — the newer version lives nowhere else). Every eviction is
// clean: the host read-back has already landed the tile's bytes, so
// pressure costs bare UNLINKs, never data. After wait() the test dumps
// trace-lanes.json plus a hand-written stats sidecar (batches, planned
// == issued, copy ops, per-lane busy ns); the pipeline test re-runs
// this binary with NNHAUL_TEST_KEEP_TRACE and cross-checks the trace
// against the sidecar.

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t kTiles = 12; // 12 chunks of working set...
constexpr std::size_t kTileBytes = 1u << 20; // ...of one chunk each
constexpr std::size_t kCapBytes = 16u << 20; // 4 whole classes: pressure

char const *const kTracePath = "trace-lanes.json";
char const *const kSidecarPath = "trace-lanes.stats.json";

bool file_exists(char const *path)
{
    std::FILE *f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        return false;
    }
    std::fclose(f);
    return true;
}

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

void delete_file(char const *path)
{
    std::remove(path);
}

bool contains(std::string const &hay, char const *needle)
{
    return hay.find(needle) != std::string::npos;
}

std::size_t count_of(std::string const &hay, char const *needle)
{
    std::size_t n = 0;
    for (std::size_t pos = hay.find(needle); pos != std::string::npos;
         pos = hay.find(needle, pos + 1))
    {
        ++n;
    }
    return n;
}

// Device write: push the pinned pattern into the tile (a 1 MiB H2D
// inside the body — real work for the device's exec lane).
void dev_write(void **buffers, void *cl_args) noexcept
{
    auto *src = *static_cast<unsigned char **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(buf->get_ptr<unsigned char>(),
        src,
        kTileBytes,
        cudaMemcpyHostToDevice,
        nnhaul::cuda_stream());
}

// Device read: pull the tile's bytes out to a pinned destination (a
// 1 MiB D2H inside the body).
void dev_read(void **buffers, void *cl_args) noexcept
{
    auto *dst = *static_cast<unsigned char **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(dst,
        buf->get_ptr<unsigned char>(),
        kTileBytes,
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
}

// Host read-back: the tile's bytes on the host lane; records the first
// byte it saw so the test proves the HOST task read the device-written
// version, not the stale host one.
struct HostSeen
{
    unsigned char *first_of_run = nullptr; // one flag per tile
    std::size_t index = 0;
};

void host_read_back(void **buffers, void *cl_args) noexcept
{
    auto *seen = static_cast<HostSeen *>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    seen->first_of_run[seen->index] = *buf->get_host_ptr<unsigned char>();
}

} // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    {
        std::cout << "skip: no CUDA device\n";
        return 0;
    }

    // The cap must come from the guard parameter: a stray
    // NNHAUL_LIMIT_CUDA_MEM in the environment would change the
    // pressure arithmetic. The trace knob is this test's own dump path
    // (overwriting whatever the outer environment set).
    unsetenv("NNHAUL_LIMIT_CUDA_MEM");
    setenv("NNHAUL_TRACE", kTracePath, 1);
    // Leftovers from an earlier run must not fake a dump.
    delete_file(kTracePath);
    delete_file(kSidecarPath);

    unsigned char *dsts = nullptr; // one 1 MiB slot per device read
    unsigned char *pattern = nullptr; // the device writes' source bytes
    if (cudaMallocHost(&dsts, kTiles * kTileBytes) != cudaSuccess ||
        cudaMallocHost(&pattern, kTileBytes) != cudaSuccess)
    {
        std::cerr << "trace-lanes: pinned host allocation failed\n";
        return 1;
    }
    std::memset(pattern, 0xAB, kTileBytes);

    {
        // The host pool must back every tile's landing: 12 chunks of
        // tiles plus the read-backs' re-landings, 256 MiB is far above.
        GpuRuntimeGuard guard(0, 1, std::size_t{256} << 20, kCapBytes);
        // Nothing preallocated: 0 grown blocks at init
        // (the cap is the ceiling; the traced run grows on demand).
        CHECK(nnhaul::stats().node[0].pool_blocks == 0);
        CHECK(nnhaul::trace_enabled());

        // Handles are neither copyable nor movable: a deque never
        // relocates its elements, so in-place emplacement is safe.
        std::deque<nnhaul::Handle> tiles;
        for (std::size_t i = 0; i < kTiles; ++i)
        {
            tiles.emplace_back(kTileBytes);
            tiles[i].set_name("lane-tile-" + std::to_string(i));
        }
        nnhaul::Codelet read_op("lane-read", nullptr, dev_read, nullptr);
        nnhaul::Codelet write_op("lane-write", nullptr, dev_write, nullptr);
        nnhaul::Codelet back_op("lane-back", host_read_back, nullptr, nullptr);

        // Host writes: every tile's version 1 lives in a host block,
        // nothing on the device — the device reads below must stage.
        // The acquires are held simultaneously so each pins its block
        // (the acquire path's window scan never evicts a live pin);
        // released in order, each publishes its host version.
        {
            std::vector<std::optional<nnhaul::Acquired>> held(kTiles);
            for (std::size_t i = 0; i < kTiles; ++i)
            {
                held[i].emplace(tiles[i].acquire(nnhaul::Access::Write));
                std::memset(held[i]->get(),
                    static_cast<int>(i + 1),
                    kTileBytes);
            }
            for (std::size_t i = 0; i < kTiles; ++i)
            {
                held[i].reset();
            }
        }

        // The pressured chain, one tile at a time: device read (staging
        // H2D), device write (the device-only version 2), host
        // read-back (the D2H that lands version 2 on the host). The
        // 12-chunk working set against the 4-block device pool forces
        // the planner to evict earlier tiles' bindings mid-batch.
        std::vector<unsigned char> first_of_run(kTiles, 0);
        for (std::size_t i = 0; i < kTiles; ++i)
        {
            unsigned char *dst = dsts + i * kTileBytes;
            HostSeen seen;
            seen.first_of_run = first_of_run.data();
            seen.index = i;
            nnhaul::insert(read_op,
                {{&tiles[i], nnhaul::Access::Read}},
                &dst,
                sizeof(dst),
                0);
            nnhaul::insert(write_op,
                {{&tiles[i], nnhaul::Access::Write}},
                &pattern,
                sizeof(pattern),
                0);
            nnhaul::insert(back_op,
                {{&tiles[i], nnhaul::Access::Read}},
                &seen,
                sizeof(seen),
                1);
        }
        nnhaul::wait();

        // The device reads saw the host-written bytes (version 1).
        for (std::size_t i = 0; i < kTiles; ++i)
        {
            unsigned char const *slot = dsts + i * kTileBytes;
            CHECK(slot[0] == static_cast<unsigned char>(i + 1));
            CHECK(slot[kTileBytes - 1] == static_cast<unsigned char>(i + 1));
        }
        // The host read-backs saw the device-written bytes (version 2).
        for (std::size_t i = 0; i < kTiles; ++i)
        {
            CHECK(first_of_run[i] == 0xAB);
        }

        // The lane counters while the runtime lives, then the dump.
        nnhaul::Stats const st = nnhaul::stats();
        nnhaul::ManagerStats const m = nnhaul::manager_stats();
        CHECK(st.lane_busy_ns.size() == 4); // exec(0), exec(1),
        // chan(0,1), chan(1,0).
        CHECK(nnhaul::trace_dump(kTracePath));

        std::ofstream side(kSidecarPath, std::ios::binary);
        side << "{\"batches\": " << m.plan_walks
             << ", \"planned_ops\": " << m.plan_ops_planned
             << ", \"issued_ops\": " << m.plan_ops_issued
             << ", \"reconcile_ok\": " << (m.plan_reconcile_ok ? "true" : "false")
             << ", \"copy_ops\": " << m.copy_ops
             << ", \"tile_ops\": " << m.tile_ops
             << ", \"cross_node_copies\": " << st.cross_node_copies
             << ", \"lane_busy_ns\": [";
        for (std::size_t l = 0; l < st.lane_busy_ns.size(); ++l)
        {
            side << (l == 0 ? "" : ", ") << st.lane_busy_ns[l];
        }
        side << "]}\n";
        side.close();

        std::cout << "trace-lanes: batches=" << m.plan_walks
                  << " planned=" << m.plan_ops_planned
                  << " issued=" << m.plan_ops_issued
                  << " copies=" << m.copy_ops
                  << " tiles=" << m.tile_ops << "\n";

        // Lane presence: the dumper emits one compact event per line,
        // so plain substring scans see every interval. Every lane of
        // the inventory carries intervals — exec(0), exec(1), and BOTH
        // channel directions (the staged hops and the read-backs).
        std::string const doc = read_file(kTracePath);
        CHECK(!doc.empty());
        CHECK(contains(doc, "\"cat\": \"exec\""));
        CHECK(contains(doc, "\"cat\": \"chan\""));
        CHECK(contains(doc, "\"lane_name\": \"exec(0)\""));
        CHECK(contains(doc, "\"lane_name\": \"exec(1)\""));
        CHECK(contains(doc, "\"lane_name\": \"chan(0,1)\""));
        CHECK(contains(doc, "\"lane_name\": \"chan(1,0)\""));
        // Kind args on the events; the ops the workload forced.
        CHECK(contains(doc, "\"kind\": \"copy\""));
        CHECK(contains(doc, "\"kind\": \"tile_op\""));
        // Both copy directions carried traffic: at least one COPY on
        // each channel lane (the metadata lanes array spells the
        // mapping lane id -> name; the events carry their lane id).
        CHECK(count_of(doc, "\"lane_name\": \"chan(1,0)\"") >= 2);
        CHECK(count_of(doc, "\"lane_name\": \"chan(0,1)\"") >= 2);

        CHECK(file_exists(kSidecarPath));
        std::string const sidecar = read_file(kSidecarPath);
        CHECK(contains(sidecar, "\"copy_ops\""));
        // The witness and the counters the sidecar carries.
        CHECK(m.plan_reconcile_ok);
        CHECK(m.plan_ops_planned == m.plan_ops_issued);
        CHECK(m.plan_walks >= 1);
        CHECK(m.copy_ops >= 2 * kTiles); // one hop in + one landing out
        // Four TILE_OPs per tile: the chain's device read, device
        // write, and host read-back, plus the seeding host-write's
        // publication op (one host TILE_OP per Write acquire).
        CHECK(m.tile_ops == 4 * kTiles);
        CHECK(st.cross_node_copies == m.copy_ops);
        // The traced busy facts: every lane carried intervals.
        for (std::size_t l = 0; l < st.lane_busy_ns.size(); ++l)
        {
            CHECK(st.lane_busy_ns[l] > 0);
        }
    }

    cudaFreeHost(dsts);
    cudaFreeHost(pattern);

    // The pipeline test keeps both files (NNHAUL_TEST_KEEP_TRACE); a
    // plain ctest run leaves nothing behind.
    if (std::getenv("NNHAUL_TEST_KEEP_TRACE") == nullptr)
    {
        delete_file(kTracePath);
        delete_file(kSidecarPath);
    }
    return g_checks_failed == 0 ? 0 : 1;
}
