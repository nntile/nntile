/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.cpp
 */

#include "nntile_context.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/context.hh>

#include <memory>
#include <mutex>

namespace torch_nntile
{

namespace
{

std::mutex g_context_mutex;
std::unique_ptr<nntile::Context> g_context;

} // namespace

void ensure_nntile_context()
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    if (g_context != nullptr)
    {
        return;
    }
    constexpr int ncpu = 1;
    constexpr int ncuda = 0;
    constexpr int ooc_enabled = 0;
    constexpr char const *ooc_path = "/tmp/nntile_ooc";
    constexpr std::size_t ooc_size = 16 * 1024 * 1024;
    constexpr int logger = 0;
    g_context = std::make_unique<nntile::Context>(
        ncpu,
        ncuda,
        ooc_enabled,
        ooc_path,
        ooc_size,
        logger);
}

} // namespace torch_nntile

#else

namespace torch_nntile
{

void ensure_nntile_context()
{
}

} // namespace torch_nntile

#endif
