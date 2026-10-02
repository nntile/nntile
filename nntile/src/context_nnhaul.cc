/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/context_nnhaul.cc
 * Context init and shutdown when libnntile is built against libnnhaul.
 */

#include "nntile/context.hh"
#include "nntile/defs.h"

#ifdef NNTILE_USE_NNHAUL

#include "nntile/backend_workers.hh"
#include "nntile/logger.hh"

#include <nnhaul/nnhaul.hh>

#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace nntile
{

int g_backend_ncpu = 0;
int g_backend_ncuda = 0;
bool g_backend_ready = false;
ScheduleKindFilter g_schedule_kind_filter = ScheduleKindFilter::Any;

std::vector<BackendWorker> backend_worker_list()
{
    std::vector<BackendWorker> workers;
    if (!g_backend_ready)
    {
        workers.push_back(BackendWorker{0, "cpu", -1});
        return workers;
    }
    for (int i = 0; i < g_backend_ncuda; ++i)
    {
        workers.push_back(BackendWorker{i, "cuda", i});
    }
    for (int i = 0; i < g_backend_ncpu; ++i)
    {
        workers.push_back(
            BackendWorker{g_backend_ncuda + i, "cpu", -1});
    }
    if (workers.empty())
    {
        workers.push_back(BackendWorker{0, "cpu", -1});
    }
    return workers;
}

namespace
{

std::size_t default_cpu_cap()
{
    long const pages = sysconf(_SC_PHYS_PAGES);
    long const page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0)
    {
        return std::size_t{1} << 30;
    }
    return static_cast<std::size_t>(pages) *
        static_cast<std::size_t>(page) / 2;
}

} // namespace

#ifdef NNTILE_USE_CUDA
cudnnHandle_t cudnn_get_local_handle()
{
    return ::nnhaul::cudnn_handle();
}
#endif // NNTILE_USE_CUDA

Context::Context(
    int ncpu,
    int ncuda,
    int ooc,
    char const *ooc_path,
    std::size_t ooc_size,
    int logger,
    char const *logger_addr,
    int logger_port,
    int verbose,
    std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each):
    initialized(0),
    ooc_disk_node_id(-1),
    verbose(verbose)
{
    (void)ooc_path;
    (void)ooc_size;
    if (ooc != 0)
    {
        throw std::runtime_error(
            "NNHaul build does not support disk out-of-core");
    }
    if (g_backend_ready)
    {
        throw std::runtime_error("NNHaul is already initialized");
    }
    if (ncpu < 0)
    {
        unsigned const hc = std::thread::hardware_concurrency();
        ncpu = hc == 0 ? 1 : static_cast<int>(hc);
    }
    if (ncpu < 1)
    {
        ncpu = 1;
    }
#ifndef NNTILE_USE_CUDA
    if (ncuda < 0)
    {
        ncuda = 0;
    }
    if (ncuda != 0)
    {
        throw std::runtime_error(
            "CPU-only NNHaul build requires ncuda == 0");
    }
    cuda_cap_bytes_each = 0;
#else
    // Worker 0 is the default CUDA worker. Callers that pass ncuda <= 0
    // still get that worker, otherwise a CUDA codelet with no hint throws.
    if (ncuda < 1)
    {
        ncuda = 1;
    }
    if (cuda_cap_bytes_each == 0)
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaError_t const err =
            cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess || free_bytes == 0)
        {
            throw std::runtime_error("cudaMemGetInfo failed");
        }
        cuda_cap_bytes_each = free_bytes - free_bytes / 10;
        if (cuda_cap_bytes_each == 0)
        {
            throw std::runtime_error("NNHaul CUDA cap is zero");
        }
    }
#endif
    if (cpu_cap_bytes == 0)
    {
        cpu_cap_bytes = default_cpu_cap();
    }
    cpu_cap_bytes_chosen = cpu_cap_bytes;
    cuda_cap_bytes_each_chosen = cuda_cap_bytes_each;
    ::nnhaul::init(
        ncpu,
        ncuda,
        cpu_cap_bytes,
        cuda_cap_bytes_each);
#ifdef NNTILE_USE_CUDA
    // CUDA codelets take this worker when insert has no worker hint.
    // The first CPU worker stays the default for CPU-only codelets.
    ::nnhaul::set_default_cuda_worker(0);
#endif
    ::nnhaul::set_default_cpu_worker(ncuda);
    g_backend_ncpu = ncpu;
    g_backend_ncuda = ncuda;
    g_backend_ready = true;
    g_schedule_kind_filter = ScheduleKindFilter::Any;
    if (logger != 0)
    {
        logger::logger_init(logger_addr, logger_port);
        if (verbose > 0)
        {
            std::cout << "Initialized logger\n";
        }
    }
    initialized = 1;
    if (verbose > 0)
    {
        std::cout << "NNTile context is initialized (NNHaul)\n";
    }
}

void Context::shutdown()
{
    if (initialized == 0)
    {
        return;
    }
    ::nnhaul::wait();
    if (logger::logger_running)
    {
        logger::logger_shutdown();
    }
    ::nnhaul::shutdown();
    g_backend_ready = false;
    g_backend_ncpu = 0;
    g_backend_ncuda = 0;
    initialized = 0;
}

void Context::restrict_cpu()
{
    g_schedule_kind_filter = ScheduleKindFilter::Cpu;
}

void Context::restrict_cuda()
{
    g_schedule_kind_filter = ScheduleKindFilter::Cuda;
}

void Context::restore_where()
{
    g_schedule_kind_filter = ScheduleKindFilter::Any;
}

} // namespace nntile

#endif // NNTILE_USE_NNHAUL
