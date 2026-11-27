/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_dtd_versions.cc
 * Version semantics on the rolling stream: R||R share one replica, W
 * excludes readers through stream order, ReadWrite retags in place.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"

#include <atomic>
#include <chrono>
#include <thread>

namespace
{

struct WriteMark
{
    int value = 0;
    std::atomic<int> *tick = nullptr;
    std::atomic<int> *mark = nullptr;
};

void write_marked(void **buffers, void *cl_args) noexcept
{
    auto *probe = static_cast<WriteMark *>(cl_args);
    if (probe->tick != nullptr && probe->mark != nullptr)
    {
        probe->mark->store(probe->tick->fetch_add(1));
    }
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(buf->get_ptr<int>(),
        &probe->value,
        sizeof(int),
        cudaMemcpyHostToDevice,
        nnhaul::cuda_stream());
}

struct ReadProbe
{
    int *value = nullptr;
    void **ptr = nullptr;
    std::atomic<int> *tick = nullptr;
    std::atomic<int> *mark = nullptr;
    bool sleep = false;
};

void read_probe(void **buffers, void *cl_args) noexcept
{
    auto *probe = static_cast<ReadProbe *>(cl_args);
    if (probe->sleep)
    {
        try
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        catch (...)
        {
        }
    }
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    int data = -1;
    cudaMemcpyAsync(&data,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
    cudaStreamSynchronize(nnhaul::cuda_stream());
    if (probe->value != nullptr)
    {
        *probe->value = data;
    }
    if (probe->ptr != nullptr)
    {
        *probe->ptr = buf->get_ptr<int>();
    }
    if (probe->tick != nullptr && probe->mark != nullptr)
    {
        probe->mark->store(probe->tick->fetch_add(1));
    }
}

struct RwProbe
{
    int *seen = nullptr;
    void **ptr = nullptr;
    int store = 0;
};

void read_write(void **buffers, void *cl_args) noexcept
{
    auto *probe = static_cast<RwProbe *>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    int data = -1;
    cudaMemcpyAsync(&data,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
    cudaStreamSynchronize(nnhaul::cuda_stream());
    *probe->seen = data;
    if (probe->ptr != nullptr)
    {
        *probe->ptr = buf->get_ptr<int>();
    }
    cudaMemcpyAsync(buf->get_ptr<int>(),
        &probe->store,
        sizeof(int),
        cudaMemcpyHostToDevice,
        nnhaul::cuda_stream());
}

void host_write(nnhaul::Handle &handle, int value)
{
    auto mem = handle.acquire(nnhaul::Access::Write);
    *mem.get_ptr<int>() = value;
}

} // namespace

// Version semantics on the rolling stream: R||R share one replica and
// pointer, W excludes readers through stream order, ReadWrite retags
// the replica in place.
int main()
{
    // The host pool needs ROOM: the version-semantics probes keep two
    // tiles live while a third streams through, and rescues of the
    // live replicas must land somewhere. With a one-block host pool
    // the world is legitimately exhausted (pool-exhaustion, the
    // plan-time fault) — not this test's subject. A 64-MiB carve
    // keeps the probe about replica versioning.
    GpuRuntimeGuard guard(0, 1, 64u << 20, nnhaul::chunk_class_bytes());
    nnhaul::Codelet write_op("write", nullptr, write_marked, nullptr);
    nnhaul::Codelet read_op("read", nullptr, read_probe, nullptr);
    nnhaul::Codelet rw_op("read-write", nullptr, read_write, nullptr);

    {
        nnhaul::Handle tile(256);
        tile.set_name("rr");
        host_write(tile, 11);
        int first = -1;
        int second = -1;
        void *p0 = nullptr;
        void *p1 = nullptr;
        ReadProbe a;
        a.value = &first;
        a.ptr = &p0;
        ReadProbe b;
        b.value = &second;
        b.ptr = &p1;
        nnhaul::insert(read_op, {{&tile, nnhaul::Access::Read}},
            &a, sizeof(a));
        nnhaul::insert(read_op, {{&tile, nnhaul::Access::Read}},
            &b, sizeof(b));
        nnhaul::wait();
        CHECK(first == 11);
        CHECK(second == 11);
        CHECK(p0 != nullptr);
        CHECK(p0 == p1);
    }

    {
        nnhaul::Handle tile(256);
        tile.set_name("rwr");
        host_write(tile, 1);
        std::atomic<int> tick{0};
        std::atomic<int> r1_end{-1};
        std::atomic<int> w_start{-1};
        int r1_value = -1;
        int r2_value = -1;
        ReadProbe r1;
        r1.value = &r1_value;
        r1.tick = &tick;
        r1.mark = &r1_end;
        r1.sleep = true;
        WriteMark write;
        write.value = 2;
        write.tick = &tick;
        write.mark = &w_start;
        ReadProbe r2;
        r2.value = &r2_value;
        nnhaul::insert(
            read_op, {{&tile, nnhaul::Access::Read}}, &r1, sizeof(r1));
        nnhaul::insert(
            write_op, {{&tile, nnhaul::Access::Write}}, &write, sizeof(write));
        nnhaul::insert(
            read_op, {{&tile, nnhaul::Access::Read}}, &r2, sizeof(r2));
        nnhaul::wait();
        CHECK(r1_value == 1);
        CHECK(r2_value == 2);
        CHECK(r1_end.load() >= 0);
        CHECK(w_start.load() > r1_end.load());
    }

    {
        nnhaul::Handle tile(256);
        tile.set_name("rw");
        host_write(tile, 5);
        int seen = -1;
        void *rw_ptr = nullptr;
        void *later_ptr = nullptr;
        int later = -1;
        RwProbe rw;
        rw.seen = &seen;
        rw.ptr = &rw_ptr;
        rw.store = 9;
        ReadProbe later_probe;
        later_probe.value = &later;
        later_probe.ptr = &later_ptr;
        nnhaul::insert(
            rw_op, {{&tile, nnhaul::Access::ReadWrite}}, &rw, sizeof(rw));
        nnhaul::insert(read_op,
            {{&tile, nnhaul::Access::Read}},
            &later_probe,
            sizeof(later_probe));
        nnhaul::wait();
        CHECK(seen == 5);
        CHECK(later == 9);
        CHECK(rw_ptr != nullptr);
        CHECK(rw_ptr == later_ptr);
    }

    return g_checks_failed == 0 ? 0 : 1;
}
