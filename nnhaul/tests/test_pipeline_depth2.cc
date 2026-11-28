/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_pipeline_depth2.cc
 * The depth-2 pipeline : a batch ISSUES
 * without joining — signal_graph_end returns while the batch's device
 * work still runs; a second batch issues over it (its host consumer
 * owing the LIVE predecessor's device op cross-lane — the late-record
 * path into a running lane); join_predecessor joins the OLDER for
 * real; wait() joins the rest; the byte results prove the ordering
 * held; and a third un-retired batch at a non-empty close is the loud
 * pipeline-depth abort.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <string>

namespace
{

// A SLOW device fill: the tile's chunks plus a long device-side sweep
// over the workspace (device work on the lane's stream — the issue
// returns immediately, the JOIN waits for it).
void slow_fill(void **buffers, void *cl_args) noexcept
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
    unsigned char *ws = static_cast<unsigned char *>(nnhaul::workspace_ptr());
    std::size_t const ws_bytes = nnhaul::workspace_bytes();
    if (ws != nullptr && ws_bytes >= (std::size_t{256} << 20))
    {
        // Over a second of DEVICE work from a few hundred enqueues:
        // the join waits for the sweep while the next batch's issue
        // (a handful of enqueues) overlaps it — the timing assertions
        // below separate the two by construction.
        for (int i = 0; i < 600; ++i)
        {
            cudaMemsetAsync(ws,
                static_cast<unsigned char>(i),
                ws_bytes - (std::size_t{64} << 20),
                nnhaul::cuda_stream());
        }
    }
}

// The host consumer of the LIVE predecessor's output (C = A + 1): its
// launch gates on the predecessor's device op — the host-lane wait of
// a device mint, with the staging COPY's wait riding the late record
// into the still-running device lane.
void host_add1(void **buffers, void *cl_args) noexcept
{
    (void) cl_args;
    auto *a = static_cast<nnhaul::Buffer *>(buffers[0]);
    auto *c = static_cast<nnhaul::Buffer *>(buffers[1]);
    unsigned int const *av = a->get_ptr<unsigned int>();
    unsigned int *cv = c->get_ptr<unsigned int>();
    std::size_t const n = c->nbytes() / sizeof(unsigned int);
    for (std::size_t i = 0; i < n; ++i)
    {
        cv[i] = av[i] + 1u;
    }
}

// The device reader: C's first int out to a host flag.
void dev_read(void **buffers, void *cl_args) noexcept
{
    auto *out = *static_cast<unsigned int **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(out,
        buf->get_ptr<unsigned int>(),
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
}

} // namespace

int main()
{
    GpuRuntimeGuard guard(0, 1, std::size_t{8} << 20, 4 * nnhaul::chunk_class_bytes());

    nnhaul::Codelet fill("pipe-fill", nullptr, slow_fill, nullptr);
    nnhaul::Codelet add1("pipe-add1", host_add1, nullptr, nullptr);
    nnhaul::Codelet reader("pipe-read", nullptr, dev_read, nullptr);

    nnhaul::Handle a(1024);
    nnhaul::Handle c(1024);
    a.set_name("a");
    c.set_name("c");
    int const pattern = 0x21;
    unsigned int seen = 0;
    unsigned int *seen_ptr = &seen;

    // ---- batch 1: the slow device write, ISSUED not joined ----
    nnhaul::insert(fill,
        {{&a, nnhaul::Access::Write}},
        &pattern,
        sizeof(pattern),
        nnhaul::cuda_worker);
    auto const t0 = std::chrono::steady_clock::now();
    nnhaul::GraphEnd end1 = nnhaul::signal_graph_end();
    auto const close1_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count();
    CHECK(end1.issued_seq == 1);
    CHECK(end1.run_pending);
    // The close is ISSUE ONLY: it must return long before the batch's
    // several-hundred-ms device sweep joins (250 ms is generous).
    CHECK(close1_ms < 250);

    // ---- batch 2: the host consumer of the LIVE predecessor ----
    nnhaul::insert(add1,
        {{&a, nnhaul::Access::Read}, {&c, nnhaul::Access::Write}},
        nullptr,
        0,
        nnhaul::host_worker());
    nnhaul::insert(reader,
        {{&c, nnhaul::Access::Read}},
        &seen_ptr,
        sizeof(seen_ptr));
    auto const t1 = std::chrono::steady_clock::now();
    nnhaul::GraphEnd end2 = nnhaul::signal_graph_end();
    auto const close2_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t1)
            .count();
    CHECK(end2.issued_seq == 2);
    CHECK(end2.run_pending);
    CHECK(close2_ms < 250);
    CHECK(nnhaul::manager_stats().plan_walks == 2);

    // ---- the join act: the OLDER batch joins for real ----
    std::uint64_t const joined = nnhaul::join_predecessor();
    CHECK(joined == 1); // two un-retired batches stood between the issues
    // Only batch 2 remains un-retired: the no-op 0.
    CHECK(nnhaul::join_predecessor() == 0);
    // The join really joined: by now batch 1's device sweep completed.
    nnhaul::wait();
    CHECK(nnhaul::manager_stats().plan_reconcile_ok);
    CHECK(nnhaul::manager_stats().plan_ops_planned ==
          nnhaul::manager_stats().plan_ops_issued);

    // ---- the ordering held: A's bytes through the cross-batch debt ----
    CHECK(seen == 0x21212122u);
    {
        auto av = a.acquire(nnhaul::Access::Read);
        CHECK(*av.get_ptr<unsigned int>() == 0x21212121u);
    }

    // ---- the loud third-batch abort (the last phase: it poisons the
    //      runtime by design, as the old engine did) ----
    {
        nnhaul::Handle x(512);
        nnhaul::Handle y(512);
        x.set_name("x");
        y.set_name("y");
        nnhaul::insert(fill, {{&x, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        (void) nnhaul::signal_graph_end();
        nnhaul::insert(fill, {{&y, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        (void) nnhaul::signal_graph_end();
        nnhaul::Handle z(512);
        z.set_name("z");
        nnhaul::insert(fill, {{&z, nnhaul::Access::Write}}, &pattern,
            sizeof(pattern), nnhaul::cuda_worker);
        bool threw = false;
        try
        {
            (void) nnhaul::signal_graph_end();
        }
        catch (nnhaul::Error const &err)
        {
            threw = true;
            CHECK(std::string(err.what()).find(
                      "pipeline depth exceeded") != std::string::npos);
        }
        CHECK(threw);
    }

    return g_checks_failed == 0 ? 0 : 1;
}
