/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_cross_batch_survival.cc
 * Cross-batch data survival: a LIVE
 * tile whose readers live in LATER batches keeps its bytes when an
 * eviction takes its last actual replica — the planner rescues onto
 * the host (the drain direction), the next batch's read refetches
 * from the rescue replica, and the bytes round-trip. A production fault
 * surfaced exactly this: a batch-scoped survival rule lost such a
 * tile.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <cuda_runtime.h>

#include <string>

namespace
{

void fill_first_int(void **buffers, void *cl_args) noexcept
{
    auto const value = *static_cast<int *>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    for (std::size_t i = 0; i < buf->chunk_count(); ++i)
    {
        cudaMemsetAsync(buf->get_chunk_ptr<void>(i),
            static_cast<unsigned char>(value),
            buf->chunk_bytes(i),
            nnhaul::cuda_stream());
    }
}

void read_first_int(void **buffers, void *cl_args) noexcept
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
    // A TWO-CHUNK device pool over a roomy host pool: batch 2's writes
    // must evict batch 1's tiles — the live ones rescued to the host.
    GpuRuntimeGuard guard(0, 1, std::size_t{16} << 20, 2 * nnhaul::chunk_class_bytes());

    nnhaul::Codelet fill("surv-fill", nullptr, fill_first_int, nullptr);
    nnhaul::Codelet reader("surv-read", nullptr, read_first_int, nullptr);

    nnhaul::Handle x(1024);
    nnhaul::Handle y(1024);
    x.set_name("x");
    y.set_name("y");
    int const vx = 0x5a;
    int const vy = 0xc3;

    // Batch 1: fill both device chunks.
    nnhaul::insert(fill, {{&x, nnhaul::Access::Write}}, &vx, sizeof(vx));
    nnhaul::insert(fill, {{&y, nnhaul::Access::Write}}, &vy, sizeof(vy));
    nnhaul::wait();
    CHECK(nnhaul::manager_stats().plan_reconcile_ok);

    // Batch 2: two more one-chunk tiles — the device free list is
    // empty, so x and y are evicted. Both are LIVE (their handles
    // stand, their readers come later): each last actual replica is
    // RESCUED to the host, never dropped.
    {
        nnhaul::Handle z(1024);
        nnhaul::Handle w(1024);
        z.set_name("z");
        w.set_name("w");
        int const vz = 0x11;
        nnhaul::insert(fill, {{&z, nnhaul::Access::Write}}, &vz, sizeof(vz));
        nnhaul::insert(fill, {{&w, nnhaul::Access::Write}}, &vz, sizeof(vz));
        nnhaul::wait();
    }

    // Batch 3: x and y are read on the device again — the reads must
    // refetch from the host rescue replicas (COPY host->device in the
    // copy log) and the bytes must be the batch-1 fills.
    int seen_x = -1;
    int seen_y = -1;
    int *px = &seen_x;
    int *py = &seen_y;
    std::size_t const copies_before = nnhaul::copy_log().size();
    nnhaul::insert(reader, {{&x, nnhaul::Access::Read}}, &px, sizeof(px));
    nnhaul::insert(reader, {{&y, nnhaul::Access::Read}}, &py, sizeof(py));
    nnhaul::wait();

    CHECK(static_cast<unsigned char>(seen_x) == 0x5a); // memset pattern
    CHECK(static_cast<unsigned char>(seen_y) == 0xc3);

    // The refetch crossed a node pair (host -> device).
    auto const log = nnhaul::copy_log();
    bool host_to_dev_refetch = false;
    for (std::size_t i = copies_before; i < log.size(); ++i)
    {
        if (log[i].src_node == 1 && log[i].dst_node == 0)
        {
            host_to_dev_refetch = true;
        }
    }
    CHECK(host_to_dev_refetch);

    // The host acquire path also sees the rescued bytes.
    {
        auto m = x.acquire(nnhaul::Access::Read);
        CHECK(static_cast<unsigned char const *>(m.get())[0] == 0x5a);
    }

    CHECK(nnhaul::manager_stats().plan_reconcile_ok);
    CHECK(nnhaul::manager_stats().plan_ops_planned ==
          nnhaul::manager_stats().plan_ops_issued);

    return g_checks_failed == 0 ? 0 : 1;
}
