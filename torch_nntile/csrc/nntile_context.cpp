/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.cpp
 */

#include "nntile_context.h"

#include "nntile_graph_recorder.h"

#include <mutex>
#include <stdexcept>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/context.hh>
#include <nntile/defs.h>

#include <starpu.h>

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
        g_context_config.verbose);
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
    bool cpu_fallback)
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
    return g_context != nullptr;
}

void ensure_nntile_context()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
}

void restrict_cpu()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
    g_context->restrict_cpu();
}

void restrict_cuda()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    create_context_locked();
    g_context->restrict_cuda();
}

void restore_where()
{
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
    if (starpu_is_initialized())
    {
        starpu_task_wait_for_all();
    }
    g_context->shutdown();
    g_context.reset();
}

} // namespace torch_nntile

#else

namespace torch_nntile
{

namespace
{

[[noreturn]] void require_libnntile()
{
    throw std::runtime_error(
        "torch_nntile context APIs require libnntile "
        "(rebuild with NNTILE_BUILD_DIR set)");
}

} // namespace

void init_context(
    int /*ncpu*/,
    int /*ncuda*/,
    int /*ooc_enabled*/,
    const char * /*ooc_path*/,
    std::size_t /*ooc_size*/,
    int /*logger*/,
    int /*verbose*/,
    bool /*cpu_fallback*/)
{
    require_libnntile();
}

bool built_with_cuda()
{
    return false;
}

bool is_cpu_fallback_enabled()
{
    return true;
}

bool is_context_initialized()
{
    return false;
}

void ensure_nntile_context()
{
}

void restrict_cpu()
{
    require_libnntile();
}

void restrict_cuda()
{
    require_libnntile();
}

void restore_where()
{
    require_libnntile();
}

void wait_for_all()
{
}

void shutdown_context()
{
}

} // namespace torch_nntile

#endif
