/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_engine_smoke.cc
 * The engine's end-to-end smoke test : a tiny CUDA fill codelet
 * and a CPU codelet over 1 GPU + host, a small mixed batch closed by
 * signal_graph_end, byte results through host acquires, the planned ==
 * issued witness, kernel launches, the per-node pool stats, and the
 * simulate exposure — the chunk-graph engine's first proof.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/plan.hh" // tile_chunk_span — the chunk span the copy
                         // log's byte stamp derives from
#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <cuda_runtime.h>

#include <cstring>
#include <string>

namespace
{

// The CUDA fill body: one byte pattern over the tile's chunks (the
// registration blob carries the pattern).
void cuda_fill(void **buffers, void *cl_args) noexcept
{
    auto const pattern =
        static_cast<unsigned char>(*static_cast<int *>(cl_args));
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemsetAsync(buf->get_chunk_ptr<void>(i),
            pattern,
            buf->chunk_bytes(i),
            nnhaul::cuda_stream());
    }
}

// The CPU body: b = a + 1 over one single-chunk tile (both operands on
// the host node's lane).
void host_scale(void **buffers, void *cl_args) noexcept
{
    (void) cl_args;
    auto *a = static_cast<nnhaul::Buffer *>(buffers[0]);
    auto *b = static_cast<nnhaul::Buffer *>(buffers[1]);
    int const *av = a->get_ptr<int>();
    int *bv = b->get_ptr<int>();
    std::size_t const n = b->nbytes() / sizeof(int);
    for (std::size_t i = 0; i < n; ++i)
    {
        bv[i] = av[i] + 1;
    }
}

// The CUDA read body: the tile's first int out to a host flag.
void cuda_read(void **buffers, void *cl_args) noexcept
{
    auto *out = *static_cast<int **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(out,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
}

} // namespace

int main()
{
    // 1 GPU + host: the device pool of 8 chunks, the host pool bound
    // of 32 MiB (8 blocks).
    GpuRuntimeGuard guard(0, 1, std::size_t{32} << 20, 8 * nnhaul::chunk_class_bytes());

    CHECK(nnhaul::host_worker() == 1);
    CHECK(nnhaul::stats().memory_nodes == 2);

    nnhaul::Codelet fill("smoke-fill", nullptr, cuda_fill, nullptr);
    nnhaul::Codelet scale("smoke-scale", host_scale, nullptr, nullptr);
    nnhaul::Codelet reader("smoke-read", nullptr, cuda_read, nullptr);

    // ---- the mixed batch: device write, host compute, device read ----
    std::size_t const tile_bytes = 1024;
    nnhaul::Handle a(tile_bytes);
    nnhaul::Handle b(tile_bytes);
    a.set_name("a");
    b.set_name("b");
    int const pattern = 0x3c;
    int seen = -1;
    int *seen_ptr = &seen;

    nnhaul::insert(fill,
        {{&a, nnhaul::Access::Write}},
        &pattern,
        sizeof(pattern),
        nnhaul::cuda_worker);
    nnhaul::insert(scale,
        {{&a, nnhaul::Access::Read}, {&b, nnhaul::Access::Write}},
        nullptr,
        0,
        nnhaul::host_worker());
    nnhaul::insert(reader,
        {{&b, nnhaul::Access::Read}},
        &seen_ptr,
        sizeof(seen_ptr));

    // The duration model loads before the first plan closes, so the
    // simulate exposure runs over the batch's own folded owed sets.
    nnhaul::set_duration(fill, 0, 1000);
    nnhaul::set_duration(scale, 1, 2000);
    nnhaul::set_duration(reader, 0, 3000);
    nnhaul::set_channel_rate(0, 1, 1); // device -> host, bytes/ns
    nnhaul::set_channel_rate(1, 0, 1); // host -> device

    // The end-of-sequence hint closes the batch (untraced runs;
    // traced runs keep the close-at-wait discipline and report the
    // pending run alone).
    nnhaul::GraphEnd end = nnhaul::signal_graph_end();
    if (!nnhaul::trace_enabled())
    {
        CHECK(end.issued_seq == 1);
    }
    else
    {
        CHECK(end.issued_seq == 0);
    }
    CHECK(end.run_pending);
    nnhaul::wait();

    // Byte results: 0x3c3c3c3c + 1 through the CPU body, back to the
    // host through the device reader.
    CHECK(seen == 0x3c3c3c3d);
    {
        auto av = a.acquire(nnhaul::Access::Read);
        CHECK(*av.get_ptr<int>() == 0x3c3c3c3c);
    }

    // The planned == issued witness and the launch counter (device
    // TILE_OPs only: the CPU scale runs on the host lane).
    nnhaul::ManagerStats m = nnhaul::manager_stats();
    CHECK(m.plan_walks == 1);
    CHECK(m.plan_ops_planned == m.plan_ops_issued);
    CHECK(m.plan_reconcile_ok);
    CHECK(m.tile_ops == 3);
    CHECK(m.copy_ops >= 2); // a: device->host, b: host->device
    CHECK(m.simulate_makespan_ns > 0);
    CHECK(nnhaul::kernel_launches() == 2);

    // Every logged transfer crossed a node pair and carried the
    // copied chunk's OCCUPIED SPAN (the same derivation the dispatch
    // and the hook use): both tiles are
    // single tail chunks of 1024 bytes — the span is the payload,
    // never the class.
    auto log = nnhaul::copy_log();
    CHECK(!log.empty());
    for (auto const &rec : log)
    {
        CHECK(rec.src_node != rec.dst_node);
        CHECK(rec.nbytes == nnhaul::tile_chunk_span(tile_bytes, 0));
    }
    CHECK(nnhaul::stats().cross_node_copies == log.size());

    // ---- a multi-chunk tile: the contiguous host acquire run ----
    {
        std::size_t const nbytes = 6 * (std::size_t{1} << 20); // 2 chunks
        nnhaul::Handle multi(nbytes);
        multi.set_name("multi");
        nnhaul::insert(fill,
            {{&multi, nnhaul::Access::Write}},
            &pattern,
            sizeof(pattern),
            nnhaul::cuda_worker);
        nnhaul::wait();
        CHECK(nnhaul::kernel_launches() == 3);
        {
            auto mem = multi.acquire(nnhaul::Access::Read);
            auto const *bytes =
                static_cast<unsigned char const *>(mem.get());
            std::size_t const cls = nnhaul::chunk_class_bytes();
            CHECK(bytes[0] == 0x3c);
            CHECK(bytes[cls - 1] == 0x3c);
            CHECK(bytes[cls] == 0x3c); // the second chunk begins here
            CHECK(bytes[nbytes - 1] == 0x3c);
        }
        multi.invalidate();
        CHECK(!nnhaul::handle_swept(multi));
        nnhaul::wait();
        CHECK(nnhaul::handle_swept(multi));
    }

    // ---- one codelet, two accesses, one tile: two symbols ----
    // The access vector is part of the symbol key: two same-shape,
    // same-args inserts of one codelet declaring DIFFERENT accesses on
    // one tile intern to two symbols and both pass the checker — a
    // single symbol would false-census the second instance with
    // access-mode-mismatch (the checker's per-(task, tile) access is
    // first-occurrence).
    {
        nnhaul::Handle t(1024);
        t.set_name("two-access");
        nnhaul::insert(fill,
            {{&t, nnhaul::Access::Write}},
            &pattern,
            sizeof(pattern),
            nnhaul::cuda_worker);
        nnhaul::wait();
        unsigned int seen2 = 0;
        unsigned int *seen2_ptr = &seen2;
        nnhaul::insert(reader,
            {{&t, nnhaul::Access::Read}},
            &seen2_ptr,
            sizeof(seen2_ptr));
        nnhaul::insert(reader,
            {{&t, nnhaul::Access::Write}},
            &seen2_ptr,
            sizeof(seen2_ptr));
        nnhaul::wait();
        CHECK(seen2 == 0x3c3c3c3c); // both instances ran, in order
        std::size_t done_peekers = 0;
        for (auto const &v : nnhaul::tasks())
        {
            if (v.phase == nnhaul::TaskPhase::Done &&
                v.codelet == "smoke-read")
            {
                done_peekers += 1;
            }
        }
        CHECK(done_peekers == 3); // batch 1's reader + the two here
    }

    // The per-node pool stats: 2 nodes, the host last, the fixed
    // device workspaces beside the pools. The blocks are
    // the GROWN count — lazily allocated on demand, at most the init
    // cap (8 device chunks here); the host bytes likewise grown, the
    // cpu_cap_bytes the chosen ceiling.
    nnhaul::Stats st = nnhaul::stats();
    CHECK(st.memory_nodes == 2);
    CHECK(st.node.size() == 2);
    CHECK(!st.node[0].host);
    CHECK(st.node[0].pool_blocks <= 8);
    CHECK(st.node[0].pool_blocks > 0);
    CHECK(st.node[0].workspace_bytes > 0);
    CHECK(st.node[1].host);
    CHECK(st.node[1].pool_bytes <= st.cpu_cap_bytes);

    // The task views carry the resolved nodes and their classes.
    std::size_t host_seen = 0;
    std::size_t cuda_seen = 0;
    for (auto const &t : nnhaul::tasks())
    {
        if (t.phase == nnhaul::TaskPhase::Done)
        {
            if (t.kind == nnhaul::Kind::Cpu)
            {
                CHECK(t.worker == 1);
                host_seen += 1;
            }
            else
            {
                CHECK(t.worker == 0);
                cuda_seen += 1;
            }
        }
    }
    CHECK(host_seen == 1);
    // fill(a), reader(b), fill(multi), fill(t), and the two reader
    // instances of the two-symbol block: six Done device tasks.
    CHECK(cuda_seen == 6);

    return g_checks_failed == 0 ? 0 : 1;
}
