/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_queue_admission.cc
 * Queue admission: submission order is execution order on the single in-
 * order compute stream.
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

std::atomic<int> g_tick{0};
std::atomic<int> g_b_end{-1};
std::atomic<int> g_c_start{-1};
std::atomic<int> g_b_value{-1};
std::atomic<int> g_c_value{-1};

// Codelet bodies run on the worker thread and enqueue onto the one
// in-order compute stream: submission order is execution order.
void slow_write(void **buffers, void *cl_args) noexcept
{
    try
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    catch (...)
    {
    }
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(buf->get_ptr<int>(),
        cl_args,
        sizeof(int),
        cudaMemcpyHostToDevice,
        nnhaul::cuda_stream());
}

void read_b(void **buffers, void *cl_args) noexcept
{
    (void) cl_args;
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    int value = -1;
    cudaMemcpyAsync(&value,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
    cudaStreamSynchronize(nnhaul::cuda_stream());
    g_b_value.store(value);
    g_b_end.store(g_tick.fetch_add(1));
}

void read_c(void **buffers, void *cl_args) noexcept
{
    (void) cl_args;
    g_c_start.store(g_tick.fetch_add(1));
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    int value = -1;
    cudaMemcpyAsync(&value,
        buf->get_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        nnhaul::cuda_stream());
    cudaStreamSynchronize(nnhaul::cuda_stream());
    g_c_value.store(value);
}

} // namespace

int main()
{
    // A 4-block host pool: cross-batch data survival keeps BOTH live
    // tiles' bytes (the rescue lands on the host), so the incidental
    // 1-block scarcity of the old config census-fires under the
    // engine-owned survival rule.
    GpuRuntimeGuard guard(0, 1, 16u << 20, nnhaul::chunk_class_bytes());
    nnhaul::Handle gate(256);
    gate.set_name("gate");
    nnhaul::Handle ready(256);
    ready.set_name("ready");

    // The fixed 1-block host pool of this scenario serves sequential
    // acquires (the engine's contiguous-run carve evicts between
    // them); stacked pins would need stacked blocks.
    {
        auto gate_mem = gate.acquire(nnhaul::Access::Write);
        *gate_mem.get_ptr<int>() = 0;
    }
    {
        auto ready_mem = ready.acquire(nnhaul::Access::Write);
        *ready_mem.get_ptr<int>() = 3;
    }

    nnhaul::Codelet write_gate("slow-write", nullptr, slow_write, nullptr);
    nnhaul::Codelet read_gate("read-gate", nullptr, read_b, nullptr);
    nnhaul::Codelet read_ready("read-ready", nullptr, read_c, nullptr);

    int seven = 7;
    nnhaul::insert(
        write_gate, {{&gate, nnhaul::Access::Write}}, &seven, sizeof(seven));
    nnhaul::insert(read_gate, {{&gate, nnhaul::Access::Read}}, nullptr, 0);
    nnhaul::insert(read_ready, {{&ready, nnhaul::Access::Read}}, nullptr, 0);
    nnhaul::wait();

    CHECK(g_b_value.load() == 7);
    CHECK(g_c_value.load() == 3);
    CHECK(g_b_end.load() >= 0);
    CHECK(g_c_start.load() > g_b_end.load());
    return g_checks_failed == 0 ? 0 : 1;
}
