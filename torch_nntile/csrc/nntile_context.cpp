/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.cpp
 * @version 2.0.0
 */

#include "nntile_context.h"

#include "nntile_graph_recorder.h"

#include <mutex>
#include <stdexcept>
#include <cstdlib>

#include <nntile/context.hh>
#include <nntile/defs.h>

#include <memory>

namespace torch_nntile
{

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
    if (ncuda > 1)
    {
        // The chunked transport is single-device; reject before locking
        // the configuration so a corrected init_context(ncuda=1) can
        // still be called in the same process.
        throw std::runtime_error(
            "torch_nntile on NNHaul supports one CUDA worker (ncuda<=1)");
    }
    if (cpu_fallback)
    {
        // The 1.x opt-in ran unregistered aten operators through the CPU
        // implementation on nntile tensors. NNTile 2.0 has no CPU compute
        // path, so accepting the flag would silently reintroduce it.
        throw std::runtime_error(
            "torch_nntile: cpu_fallback is not supported; unregistered "
            "operators raise. Move tensors with .to(\"nntile\") / .to(\"cpu\") "
            "and use the classic nntile operators");
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

void wait_for_all()
{
    // Join the engine for host-visible completion (reclaim already ran at
    // run()).
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
    g_context->shutdown();
    g_context.reset();
}

} // namespace torch_nntile
