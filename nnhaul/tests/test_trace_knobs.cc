/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_trace_knobs.cc
 * NNHAUL_TRACE knob lifecycle: enable/off semantics per init cycle,
 * the shutdown dump's structure, and mid-run trace_dump growth — over
 * the inventory-derived lane set of the new tracer (format_version 1:
 * one thread_name per LANE, lane args on every X event, the plan
 * metadata's planned == issued witness; the old three-stream constants
 * and the C memory counters died with the reactive engine, and the
 * dump carries no codelet names, so the old escaping checks have no
 * subject left).
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

// NNHAUL_TRACE lifecycle: each knob value gets its own
// init/tiny-workload/shutdown cycle (the knob is read per-init, like
// NNHAUL_LIMIT_CUDA_MEM). The checks are substring/structural only —
// no JSON parser in C++ here; the Python tools own real parsing. The
// mid-run growth check drives REAL workloads through the public API so
// the stamps, fires, and harvests all run.

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{

// k = 1: device node 0, host node 1 — FOUR lanes: exec(0), exec(1),
// chan(0,1), chan(1,0) (device<->host both directions always).
constexpr int kLaneCount = 4;

void set_env(char const *name, char const *value)
{
    setenv(name, value, 1);
}

void unset_env(char const *name)
{
    unsetenv(name);
}

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

bool not_contains(std::string const &hay, char const *needle)
{
    return hay.find(needle) == std::string::npos;
}

// Extract the integer following `"key": ` — the dumper writes plain
// decimal uint64s at a fixed shape, so no JSON parser is needed.
std::uint64_t u64_after_key(std::string const &doc, char const *key)
{
    std::string const marker = std::string("\"") + key + "\": ";
    std::size_t const at = doc.find(marker);
    if (at == std::string::npos)
    {
        return 0;
    }
    return std::strtoull(doc.c_str() + at + marker.size(), nullptr, 10);
}

void noop_cuda(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    (void) cl_args;
}

void noop_cpu(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    (void) cl_args;
}

// One device task plus one host task over one tile: enough work for a
// full init/execute/shutdown cycle per knob value, touching an exec
// lane of each class.
void run_tiny_workload()
{
    nnhaul::Handle tile(256);
    tile.set_name("trace-knobs-tile");
    nnhaul::Codelet touch("trace-touch", noop_cpu, noop_cuda, nullptr);
    nnhaul::insert(touch, {{&tile, nnhaul::Access::Write}}, nullptr, 0, 0);
    nnhaul::insert(
        touch, {{&tile, nnhaul::Access::ReadWrite}}, nullptr, 0, 1);
    nnhaul::wait();
}

// A cross-node workload over one host-seeded tile: the device READS it
// (a planned COPY host->device on chan(1,0)), WRITES it (a newer
// device-only version), and a host task reads THAT back (COPY
// device->host on chan(0,1)) — all four lanes of the inventory carry
// an interval, both event cats appear.
void run_mixed_workload()
{
    nnhaul::Handle tile(std::size_t{1} << 20);
    tile.set_name("trace-knobs-cross");
    nnhaul::Codelet touch("trace-touch", noop_cpu, noop_cuda, nullptr);
    {
        auto held = tile.acquire(nnhaul::Access::Write);
        std::memset(held.get(), 1, tile.nbytes());
    }
    nnhaul::insert(touch, {{&tile, nnhaul::Access::Read}}, nullptr, 0, 0);
    nnhaul::insert(touch, {{&tile, nnhaul::Access::Write}}, nullptr, 0, 0);
    nnhaul::insert(
        touch, {{&tile, nnhaul::Access::Read}}, nullptr, 0, 1);
    nnhaul::wait();
}

// Structural checks every dump must pass: JSON object shape, the
// always-present lane/process metadata events, per-lane thread names
// (the FULL frame: an empty lane is a finding), the lane args, and the
// format-1 metadata — with the deleted machinery (C counters, cap and
// policy keys) asserting ABSENT.
void check_trace_structure(std::string const &doc)
{
    CHECK(!doc.empty());
    CHECK(doc.front() == '{');
    CHECK(doc.substr(doc.size() < 2 ? 0 : doc.size() - 2) == "}\n");
    CHECK(contains(doc, "\"traceEvents\""));
    CHECK(contains(doc, "\"process_name\""));
    CHECK(contains(doc, "\"thread_name\""));
    // One thread_name per lane of the inventory, named by lane_name —
    // and no phantom lanes: with k = 1 the host is node 1, so
    // chan(0,1)/chan(1,0) are the whole channel set.
    CHECK(contains(doc, "\"exec(0)\""));
    CHECK(contains(doc, "\"exec(1)\""));
    CHECK(contains(doc, "\"chan(0,1)\""));
    CHECK(contains(doc, "\"chan(1,0)\""));
    CHECK(not_contains(doc, "chan(0,2)"));
    CHECK(not_contains(doc, "chan(2,0)"));
    // Format 1's metadata block.
    CHECK(contains(doc, "\"format_version\": 1"));
    CHECK(contains(doc, "\"runtime_version\""));
    CHECK(contains(doc, "\"lane_count\""));
    CHECK(u64_after_key(doc, "lane_count") ==
          static_cast<std::uint64_t>(kLaneCount));
    CHECK(contains(doc, "\"wall_clock_span_us\""));
    CHECK(contains(doc, "\"plan\""));
    CHECK(contains(doc, "\"reconcile_ok\""));
    // The old engine's counter events and cap/policy metadata are
    // deleted machinery: they must not resurface.
    CHECK(not_contains(doc, "\"ph\": \"C\""));
    CHECK(not_contains(doc, "vram_bytes"));
    CHECK(not_contains(doc, "ram_pinned_bytes"));
    CHECK(not_contains(doc, "vram_cap_bytes"));
    CHECK(not_contains(doc, "offload_policy"));
    CHECK(not_contains(doc, "gpu-compute"));
    CHECK(not_contains(doc, "h2d-bus"));
    CHECK(not_contains(doc, "d2h-bus"));
}

} // namespace

int main()
{
    // Leftovers from an earlier run must not fake a dump.
    delete_file("nnhaul-trace.json");

    // ---- unset: tracing fully off, no file anywhere ----
    unset_env("NNHAUL_TRACE");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(!nnhaul::trace_enabled());
        CHECK(!nnhaul::trace_dump("trace-knobs-unset-dump.json"));
    }
    CHECK(!file_exists("nnhaul-trace.json"));
    CHECK(!file_exists("trace-knobs-unset-dump.json"));

    // ---- "1": default path in the process CWD at shutdown ----
    set_env("NNHAUL_TRACE", "1");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(nnhaul::trace_enabled());
    }
    CHECK(file_exists("nnhaul-trace.json"));
    check_trace_structure(read_file("nnhaul-trace.json"));
    delete_file("nnhaul-trace.json");

    // ---- custom path: written there verbatim, nowhere else ----
    set_env("NNHAUL_TRACE", "trace-knobs-custom.json");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(nnhaul::trace_enabled());
    }
    CHECK(file_exists("trace-knobs-custom.json"));
    check_trace_structure(read_file("trace-knobs-custom.json"));
    CHECK(!file_exists("nnhaul-trace.json"));
    delete_file("trace-knobs-custom.json");

    // ---- "0": off, no file ----
    set_env("NNHAUL_TRACE", "0");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(!nnhaul::trace_enabled());
        CHECK(!nnhaul::trace_dump("trace-knobs-zero-dump.json"));
    }
    CHECK(!file_exists("nnhaul-trace.json"));
    CHECK(!file_exists("trace-knobs-zero-dump.json"));

    // ---- "" (empty): stderr warning + off, no file ----
    set_env("NNHAUL_TRACE", "");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(!nnhaul::trace_enabled());
    }
    CHECK(!file_exists("nnhaul-trace.json"));

    // ---- mid-run trace_dump: parseable, and it grows with real
    // batches. A tiny batch first, then a cross-node batch so every
    // lane of the inventory carries an interval and both cats appear.
    set_env("NNHAUL_TRACE", "1");
    {
        GpuRuntimeGuard guard(
            0, 1, std::size_t{32} << 20, 2 * nnhaul::chunk_class_bytes());
        run_tiny_workload();
        CHECK(nnhaul::trace_dump("trace-knobs-mid1.json"));
        run_mixed_workload();
        CHECK(nnhaul::trace_dump("trace-knobs-mid2.json"));
    }
    {
        std::string const mid1 = read_file("trace-knobs-mid1.json");
        std::string const mid2 = read_file("trace-knobs-mid2.json");
        check_trace_structure(mid1);
        check_trace_structure(mid2);
        // Metadata is fixed-size; only the event array grows.
        CHECK(mid2.size() > mid1.size());
        CHECK(contains(mid2, "\"ph\": \"X\""));
        CHECK(contains(mid2, "\"cat\": \"exec\""));
        CHECK(contains(mid2, "\"cat\": \"chan\""));
        CHECK(contains(mid2, "\"kind\": \"copy\""));
        CHECK(contains(mid2, "\"kind\": \"tile_op\""));
        // The lane fields, both spellings, on the events.
        CHECK(contains(mid2, "\"lane\": "));
        CHECK(contains(mid2, "\"lane_name\": \"exec(0)\""));
        CHECK(contains(mid2, "\"lane_name\": \"exec(1)\""));
        CHECK(contains(mid2, "\"lane_name\": \"chan(1,0)\""));
        CHECK(contains(mid2, "\"lane_name\": \"chan(0,1)\""));
        // Real stamps anchor the span: the batches span nonzero wall
        // time, and the second dump's span does not shrink.
        CHECK(u64_after_key(mid1, "wall_clock_span_us") > 0);
        CHECK(u64_after_key(mid2, "wall_clock_span_us") >=
              u64_after_key(mid1, "wall_clock_span_us"));
        // The plan metadata block: batches counted, ops issued,
        // planned == issued (the witness a critic verifies from the
        // file alone).
        CHECK(u64_after_key(mid1, "batches") >= 1);
        CHECK(u64_after_key(mid2, "batches") >
              u64_after_key(mid1, "batches"));
        CHECK(contains(mid2, "\"reconcile_ok\": true"));
        CHECK(u64_after_key(mid2, "planned_ops") >=
              u64_after_key(mid1, "planned_ops"));
    }
    delete_file("trace-knobs-mid1.json");
    delete_file("trace-knobs-mid2.json");
    delete_file("nnhaul-trace.json"); // the shutdown dump of this cycle

    unset_env("NNHAUL_TRACE");
    return g_checks_failed == 0 ? 0 : 1;
}
