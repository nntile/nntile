/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/context.cc
 * Context init and shutdown on the NNHaul runtime.
 * @version 2.0.0
 */

#include "nntile/context.hh"
#include "nntile/defs.h"


#include "nntile/backend_workers.hh"
#include "nntile/logger.hh"

#include <cstdio>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <nnhaul/nnhaul.hh>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace nntile
{

int g_backend_ncuda = 0;
bool g_backend_ready = false;

std::vector<BackendWorker> backend_worker_list()
{
    // libnnhaul is GPU-only: it creates exactly one CUDA worker and no
    // CPU workers. Never advertise a CPU worker here -- the scheduler
    // would assign tiles to it and the hint would then be remapped,
    // which is exactly the phantom-CPU-worker bug this list prevents.
    std::vector<BackendWorker> workers;
    int const n = g_backend_ready ? g_backend_ncuda : 0;
    if (n <= 0)
    {
        workers.push_back(BackendWorker{0, "cuda", 0});
        return workers;
    }
    for (int i = 0; i < n; ++i)
    {
        workers.push_back(BackendWorker{i, "cuda", i});
    }
    return workers;
}

#ifdef NNTILE_USE_CUDA
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
    verbose(verbose)
{
    (void)ooc_path;
    (void)ooc_size;
    if (ooc != 0)
    {
        throw std::runtime_error(
            "NNTile 2.0 does not support disk out-of-core");
    }
    if (std::getenv("NNHAUL_DEVICES") != nullptr)
    {
        throw std::runtime_error(
            "NNTile 2.0 runs on one CUDA device selected by "
            "CUDA_VISIBLE_DEVICES; NNHAUL_DEVICES is not supported");
    }
    if (g_backend_ready)
    {
        throw std::runtime_error("NNHaul is already initialized");
    }
    // libnnhaul is GPU-only: it creates one CUDA worker and no CPU
    // workers. ``ncpu`` is accepted for source compatibility but is NOT
    // recorded as a live worker -- advertising it would make the
    // scheduler split tiles onto a CPU worker that does not exist.
    // Worker 0 is the default CUDA worker. Callers that pass ncuda <= 0
    // still get that worker, otherwise a CUDA codelet with no hint throws.
    (void) ncpu;
    ncpu = 0;
    if (ncuda < 1)
    {
        ncuda = 1;
    }
    if (ncuda > 1)
    {
        throw std::runtime_error(
            "NNTile on NNHaul supports one CUDA worker: the chunked tile "
            "transport is single-device. Run one process per GPU (ncuda=1).");
    }
    if (cuda_cap_bytes_each == 0
        && std::getenv("NNHAUL_LIMIT_CUDA_MEM") == nullptr)
    {
        cudaError_t const dev_err = cudaSetDevice(0);
        if (dev_err != cudaSuccess)
        {
            throw std::runtime_error(
                "NNTile: cudaSetDevice(0) failed; select the GPU with "
                "CUDA_VISIBLE_DEVICES");
        }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaError_t const err =
            cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess || free_bytes == 0)
        {
            throw std::runtime_error("cudaMemGetInfo failed");
        }
        // The chunks live in the arena; the fixed per-device
        // workspaces (the GEMM gather/scatter workspace, the cuBLAS
        // workspace, the copy-descriptor scratch) live BESIDE it and
        // are claimed after it. The default chunk budget is therefore
        // 90% of the free VRAM MINUS the workspace size, so the arena
        // can never starve those cudaMallocs (before this, the 90%
        // arena left an uncapped init failing on its own workspace or
        // on nntile's 8-MiB chunk-pointer table).
        std::size_t workspace_bytes = std::size_t{1024} << 20;
        if (char const *ws = std::getenv("NNHAUL_WORKSPACE_MB"))
        {
            long const mib = std::strtol(ws, nullptr, 10);
            if (mib > 0)
            {
                workspace_bytes = static_cast<std::size_t>(mib) << 20;
            }
        }
        std::size_t const ninety = free_bytes - free_bytes / 10;
        cuda_cap_bytes_each =
            ninety > workspace_bytes ? ninety - workspace_bytes : 0;
        if (cuda_cap_bytes_each == 0)
        {
            throw std::runtime_error("NNHaul CUDA cap is zero");
        }
    }
    // Leave cpu_cap_bytes == 0 when the caller named none: nnhaul reads
    // 0 as "unlimited", i.e. the host pool grows only on demand and has
    // no soft wall to seek. Do NOT substitute a fraction of physical RAM
    // here -- nnhaul's planning-time growth fills a finite limit to the
    // wall and faults in every page, which on a large host turns the
    // first host-pool acquire into an allocation of half the machine.
    // A caller that wants a ceiling passes cpu_cap_bytes explicitly or
    // sets NNHAUL_LIMIT_CPU_MEM.
#if defined(__unix__) || defined(__APPLE__)
    {
        // A preloaded copy of nnhaul (e.g. a plugin linked against the
        // static archive with RTLD_GLOBAL) could satisfy these symbols
        // with a second engine. Require the DSO that actually defines
        // nnhaul::init to be an nnhaul shared library; packaged names
        // such as libnnhaul-<hash>.so (auditwheel) are accepted. A
        // deliberately renamed preload is outside this guarantee.
        Dl_info info{};
        if(dladdr(reinterpret_cast<void *>(&::nnhaul::init), &info) == 0
            || info.dli_fname == nullptr)
        {
            throw std::runtime_error(
                "nntile: cannot identify the DSO providing NNHaul");
        }
        std::string const dso(info.dli_fname);
        std::string const base = dso.substr(dso.find_last_of('/') + 1);
        if(base.rfind("libnnhaul", 0) != 0)
        {
            throw std::runtime_error(
                "nntile: NNHaul symbols resolve to '" + dso
                + "', not an nnhaul shared library; a preloaded copy of "
                  "the engine would run a second instance");
        }
    }
#endif
    ::nnhaul::init(
        ncpu,
        ncuda,
        cpu_cap_bytes,
        cuda_cap_bytes_each);
#ifdef NNTILE_USE_CUDA
    // CUDA codelets take this worker when insert has no worker hint.
    ::nnhaul::set_default_cuda_worker(0);
#endif
    // No CPU worker exists in the runtime, so none may appear in the
    // backend list or the schedule (see backend_worker_list above).
    g_backend_ncuda = ncuda;
    g_backend_ready = true;
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
        std::cout << "NNTile context is initialized (NNHaul: ncuda="
                  << g_backend_ncuda << ")\n";
    }
}

void Context::shutdown()
{
    if (initialized == 0)
    {
        return;
    }
    try
    {
        ::nnhaul::wait();
    }
    catch (std::exception const &e)
    {
        // Teardown must not rethrow from a destructor path; the fault
        // was already latched by the engine and is reported here.
        std::fprintf(stderr, "nntile: NNHaul teardown: %s\n", e.what());
    }
    if (logger::logger_running)
    {
        logger::logger_shutdown();
    }
    ::nnhaul::shutdown();
    g_backend_ready = false;
    g_backend_ncuda = 0;
    initialized = 0;
}

} // namespace nntile
