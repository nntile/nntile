/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file tests/test_default_worker.cc
 * Worker resolution over the node inventory: worker_unspecified resolves
 * to a default — a CUDA-body codelet to the device-class default (device
 * node 0), a CPU-only codelet to the HOST node, where it RUNS (the old
 * engine's CPU dead end is superseded); the setters report loud errors;
 * the explicit id range check 0..k; and insert-never-censuses (a
 * CUDA-body codelet
 * named for the host node submits fine — task-not-runnable is the
 * checker's census at wait).
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "check.hh"
#include "nnhaul/test_hooks.hh"

#include <atomic>
#include <string>

namespace
{

struct Where
{
    std::atomic<int> *worker = nullptr;
    std::atomic<int> *kind = nullptr;
    std::atomic<int> *device = nullptr;
};

void cpu_mark(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    auto *where = static_cast<Where *>(cl_args);
    where->worker->store(nnhaul::current_worker());
    where->kind->store(static_cast<int>(nnhaul::current_kind()));
}

void cuda_mark(void **buffers, void *cl_args) noexcept
{
    (void) buffers;
    auto *where = static_cast<Where *>(cl_args);
    where->worker->store(nnhaul::current_worker());
    where->kind->store(static_cast<int>(nnhaul::current_kind()));
    if (where->device != nullptr)
    {
        where->device->store(nnhaul::device_index());
    }
}

// k = 1: device node 0, host node 1 (WorkerId IS NodeId by identity).
// A CUDA-body codelet's worker_unspecified default is device node 0
// (cuda_worker, absent the setter); a CPU-only codelet's default is the
// host node — and the body really runs on the host worker's lane.
int run_one_device_runtime()
{
    GpuRuntimeGuard guard(0, 1, 1u << 20, 4 * nnhaul::chunk_class_bytes());
    CHECK(nnhaul::host_worker() == 1);
    CHECK(nnhaul::stats().memory_nodes == 2);

    nnhaul::Handle tile(256);
    tile.set_name("default-worker-tile");
    nnhaul::Codelet both("both", cpu_mark, cuda_mark, nullptr);
    nnhaul::Codelet cpu_only("cpu-only", cpu_mark, nullptr, nullptr);
    nnhaul::Codelet cuda_only("cuda-only", nullptr, cuda_mark, nullptr);
    nnhaul::Codelet nowhere("nowhere", nullptr, nullptr, nullptr);

    std::atomic<int> worker{-2};
    std::atomic<int> kind{-1};
    std::atomic<int> device{-2};
    Where where;
    where.worker = &worker;
    where.kind = &kind;
    where.device = &device;

    // An unspecified worker on a CUDA-body codelet resolves to the
    // device-class default — device node 0 — and runs there.
    nnhaul::insert(both, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where));
    nnhaul::wait();
    CHECK(worker.load() == 0);
    CHECK(kind.load() == static_cast<int>(nnhaul::Kind::Cuda));
    CHECK(device.load() == 0);

    // The replacement for the old CPU dead end: a CPU-only
    // codelet resolves to the HOST node and runs (Kind labels it, the
    // host lane executes it — kernel_launches counts device bodies
    // only, so it does not move).
    worker.store(-2);
    kind.store(-1);
    std::uint64_t const launches_before = nnhaul::kernel_launches();
    nnhaul::insert(
        cpu_only, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where));
    nnhaul::wait();
    CHECK(worker.load() == 1);
    CHECK(kind.load() == static_cast<int>(nnhaul::Kind::Cpu));
    CHECK(nnhaul::kernel_launches() == launches_before);

    // The resolution is recorded at submit and carried into the task
    // views verbatim .
    std::size_t cpu_views = 0;
    std::size_t cuda_views = 0;
    for (auto const &v : nnhaul::tasks())
    {
        if (v.phase != nnhaul::TaskPhase::Done)
        {
            continue;
        }
        if (v.codelet == "cpu-only")
        {
            CHECK(v.worker == 1);
            CHECK(v.kind == nnhaul::Kind::Cpu);
            cpu_views += 1;
        }
        else if (v.codelet == "both")
        {
            CHECK(v.worker == 0);
            CHECK(v.kind == nnhaul::Kind::Cuda);
            cuda_views += 1;
        }
    }
    CHECK(cpu_views == 1);
    CHECK(cuda_views == 1);

    // ---- the setters : loud on every wrong class ----

    nnhaul::set_default_cuda_worker(0); // the one device node: fine.
    nnhaul::set_default_cpu_worker(nnhaul::host_worker()); // the host.

    bool threw = false;
    try
    {
        nnhaul::set_default_cuda_worker(nnhaul::host_worker());
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("must name a DEVICE node") !=
              std::string::npos);
    }
    CHECK(threw);

    threw = false;
    try
    {
        nnhaul::set_default_cpu_worker(0);
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("must name the HOST node") !=
              std::string::npos);
    }
    CHECK(threw);

    // ---- the explicit id range check  ----

    threw = false;
    try
    {
        nnhaul::insert(
            both, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where), 2);
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("unknown worker id") !=
              std::string::npos);
    }
    CHECK(threw);

    threw = false;
    try
    {
        nnhaul::insert(
            both, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where), -2);
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("unknown worker id") !=
              std::string::npos);
    }
    CHECK(threw);

    // ---- a codelet with neither body: no runnable node exists, the
    //      caller-defect insert reports it loudly (never a census). ----

    threw = false;
    try
    {
        nnhaul::insert(
            nowhere, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where));
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("nowhere") != std::string::npos);
        CHECK(std::string(err.what()).find("neither a CUDA nor a CPU") !=
              std::string::npos);
    }
    CHECK(threw);

    // The runtime never entered the error state: work still runs.
    worker.store(-2);
    kind.store(-1);
    nnhaul::insert(
        cuda_only, {{&tile, nnhaul::Access::Write}}, &where, sizeof(where));
    nnhaul::wait();
    CHECK(worker.load() == 0);
    CHECK(kind.load() == static_cast<int>(nnhaul::Kind::Cuda));
    CHECK(nnhaul::kernel_launches() == launches_before + 1);

    // ---- insert never censuses : a CUDA-body codelet
    //      explicitly named for the HOST node submits fine; the fault
    //      is the checker's task-not-runnable, censused at wait. ----

    threw = false;
    try
    {
        nnhaul::insert(cuda_only,
            {{&tile, nnhaul::Access::Write}},
            &where,
            sizeof(where),
            nnhaul::host_worker());
    }
    catch (nnhaul::Error const &)
    {
        threw = true; // the submit itself must stay silent.
    }
    CHECK(!threw);
    threw = false;
    try
    {
        nnhaul::wait();
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("task-not-runnable") !=
              std::string::npos);
    }
    CHECK(threw);

    return g_checks_failed == 0 ? 0 : 1;
}

// host_worker() before init is loud — k is runtime inventory truth
//  — and so are the setters.
void check_before_init_is_loud()
{
    bool threw = false;
    try
    {
        (void) nnhaul::host_worker();
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("before nnhaul::init") !=
              std::string::npos);
    }
    CHECK(threw);

    threw = false;
    try
    {
        nnhaul::set_default_cuda_worker(0);
    }
    catch (nnhaul::Error const &err)
    {
        threw = true;
        CHECK(std::string(err.what()).find("before nnhaul::init") !=
              std::string::npos);
    }
    CHECK(threw);
}

} // namespace

int main()
{
    check_before_init_is_loud();
    return run_one_device_runtime();
}
