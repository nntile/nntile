/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.cpp
 */

#include "nntile_context.h"

#include "nntile_graph_recorder.h"

#include <mutex>
#include <stdexcept>
#include <cstdlib>

#include <nntile/context.hh>
#include <nntile/defs.h>

#ifdef NNTILE_USE_NNHAUL
#include <nntile/nnhaul/chunk_gate.hh>
#endif

#ifndef NNTILE_USE_NNHAUL
#include <starpu.h>
#endif

#include <memory>

namespace torch_nntile
{

void apply_torch_session_chunk_policy()
{
#ifdef NNTILE_USE_NNHAUL
    // A torch_nntile session may lower torch_dispatch ops (TILE_TORCH_*),
    // which consume tiles through contiguous torch tensors and cannot
    // read a chunk chain — the process-wide session law in
    // nntile/nnhaul/chunk_gate.hh. Flip the chunked-tile gate OFF at
    // session start, BEFORE the first tile allocates: the gate flip
    // inside Runtime::compile only affects future allocations, so a
    // session that allocates tiles before its first torch-op graph
    // compiles would hit the compile-time guard even with that flip
    // working as designed (torch_nntile sessions always mix classic and
    // torch ops on the same tiles, so chunking must be off for the
    // WHOLE session). An explicitly set NNHAUL_CHUNK_TILES pins the
    // session's choice (=0 unchunked from the start, =1 chunked for
    // classic-only sessions that never record torch ops — a torch-op
    // graph plus chunked tiles still fails loudly at compile); the
    // default (unset) is unchunked for torch_nntile.
    if (std::getenv("NNHAUL_CHUNK_TILES") == nullptr
        && ::nntile::haul::tile_chunking_enabled())
    {
        ::nntile::haul::set_tile_chunking(false);
    }
#endif
}

bool built_with_cuda()
{
#ifdef NNTILE_USE_CUDA
    return true;
#else
    return false;
#endif
}

namespace
{

struct ContextConfig
{
    int ncpu = 1;
    int ncuda = 0;
    int ooc_enabled = 0;
    const char *ooc_path = "/tmp/nntile_ooc";
    std::size_t ooc_size = 16 * 1024 * 1024;
    int logger = 0;
    int verbose = 0;
    bool cpu_fallback = false;
    std::size_t cpu_cap_bytes = 0;
    std::size_t cuda_cap_bytes_each = 0;
};

std::mutex g_context_mutex;
std::unique_ptr<nntile::Context> g_context;
ContextConfig g_context_config;
bool g_context_config_locked = false;

void create_context_locked()
{
    if (g_context != nullptr)
    {
        return;
    }
    g_context_config_locked = true;
    apply_torch_session_chunk_policy();
    g_context = std::make_unique<nntile::Context>(
        g_context_config.ncpu,
        g_context_config.ncuda,
        g_context_config.ooc_enabled,
        g_context_config.ooc_path,
        g_context_config.ooc_size,
        g_context_config.logger,
        "localhost",
        5001,
        g_context_config.verbose,
        g_context_config.cpu_cap_bytes,
        g_context_config.cuda_cap_bytes_each);
}

} // namespace

void init_context(
    int ncpu,
    int ncuda,
    int ooc_enabled,
    const char *ooc_path,
    std::size_t ooc_size,
    int logger,
    int verbose,
    bool cpu_fallback,
    std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each)
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    if (g_context != nullptr)
    {
        throw std::runtime_error(
            "torch_nntile.init_context() must be called before "
            "any nntile operation");
    }
    if (g_context_config_locked)
    {
        throw std::runtime_error(
            "torch_nntile context configuration is already locked");
    }
    g_context_config.ncpu = ncpu;
    g_context_config.ncuda = ncuda;
    g_context_config.ooc_enabled = ooc_enabled;
    g_context_config.ooc_path = ooc_path;
    g_context_config.ooc_size = ooc_size;
    g_context_config.logger = logger;
    g_context_config.verbose = verbose;
    g_context_config.cpu_fallback = cpu_fallback;
    g_context_config.cpu_cap_bytes = cpu_cap_bytes;
    g_context_config.cuda_cap_bytes_each = cuda_cap_bytes_each;
    g_context_config_locked = true;
}

bool is_cpu_fallback_enabled()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    return g_context_config.cpu_fallback;
}

bool is_context_initialized()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    return g_context != nullptr || platform_session_active();
}

void ensure_nntile_context()
{
    if (platform_session_active())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
}

bool uses_nnhaul()
{
#ifdef NNTILE_USE_NNHAUL
    return true;
#else
    return false;
#endif
}

void restrict_cpu()
{
    if (platform_session_active())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
    g_context->restrict_cpu();
}

void restrict_cuda()
{
    if (platform_session_active())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
    g_context->restrict_cuda();
}

void restore_where()
{
    if (platform_session_active())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
    g_context->restore_where();
}

void wait_for_all()
{
    // Join StarPU for host-visible completion (reclaim already ran at run()).
    wait_graph_session();
}

void shutdown_context()
{
    shutdown_recorder();
    std::lock_guard<std::mutex> lock(g_context_mutex);
    if (g_context == nullptr)
    {
        return;
    }
#ifndef NNTILE_USE_NNHAUL
    if (starpu_is_initialized())
    {
        starpu_task_wait_for_all();
    }
#endif
    g_context->shutdown();
    g_context.reset();
}

} // namespace torch_nntile

