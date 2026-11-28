/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_cpu_gpu_copy.cc
 * Host-written bytes survive the round trip through GPU replicas (host
 * write, kernel read back).
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cuda_runtime.h>
#include "check.hh"
#include "nnhaul/test_hooks.hh"

namespace
{

struct Probe
{
    int value = 0;
    void **ptr = nullptr;
    int *seen = nullptr;
};

void gpu_read(void **buffers, void *cl_args) noexcept
{
    auto *seen = *static_cast<int **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(seen,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
}

} // namespace

unsigned char *nnhaul_diag_gpu_chunk0();
static unsigned char *g_diag = nullptr;
unsigned char *nnhaul_diag_gpu_chunk0() { return g_diag; }
int main()
{
    // A 4-block host pool: cross-batch data survival keeps BOTH live
    // tiles' bytes (the rescue lands on the host), so the incidental
    // 1-block scarcity of the old config census-fires under the
    // engine-owned survival rule.
    GpuRuntimeGuard guard(0, 1, 16u << 20, nnhaul::chunk_class_bytes());
    nnhaul::Codelet gpu_op("gpu-read", nullptr, gpu_read, nullptr);
    nnhaul::Handle tile(4096);
    tile.set_name("h2d");
    {
        auto mem = tile.acquire(nnhaul::Access::Write);
        *mem.get_ptr<int>() = 42;
    }
    int gpu_seen = -1;
    int *gpu_seen_ptr = &gpu_seen;
    nnhaul::insert(gpu_op,
        {{&tile, nnhaul::Access::Read}},
        &gpu_seen_ptr,
        sizeof(gpu_seen_ptr));
    nnhaul::wait();
    CHECK(gpu_seen == 42);
    auto log = nnhaul::copy_log();
    CHECK(!log.empty());
    for (auto const &rec : log)
    {
        CHECK(rec.src_node != rec.dst_node);
        CHECK(rec.handle == "h2d");
    }
    CHECK(nnhaul::stats().cross_node_copies == log.size());

    // Unregister while the H2D copy is still queued must not drop the
    // source out from under the CUDA reader. That is the host-read path.
    nnhaul::Handle late(4096);
    late.set_name("late");
    {
        auto mem = late.acquire(nnhaul::Access::Write);
        *mem.get_ptr<int>() = 7;
    }
    int late_seen = -1;
    int *late_seen_ptr = &late_seen;
    nnhaul::insert(gpu_op,
        {{&late, nnhaul::Access::Read}},
        &late_seen_ptr,
        sizeof(late_seen_ptr));
    late.invalidate();
    nnhaul::wait();
    std::fprintf(stderr, "[diag] late_seen=%d\n", late_seen);
    CHECK(late_seen == 7);
    return g_checks_failed == 0 ? 0 : 1;
}
