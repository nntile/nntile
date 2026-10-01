/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/sync_defer.hh
 * Same TLS defer flag as the StarPU build. Wait calls ``nnhaul::wait``.
 */

#pragma once

#include <nnhaul/nnhaul.hh>

#include <atomic>
#include <cstdint>

namespace nntile
{

extern thread_local int g_starpu_sync_defer_depth;

extern std::atomic<std::uint64_t> g_starpu_wait_for_all_count;

inline void starpu_task_wait_for_all_counted()
{
    ++g_starpu_wait_for_all_count;
    ::nnhaul::wait();
}

struct StarpuSyncDefer
{
    StarpuSyncDefer()
    {
        ++g_starpu_sync_defer_depth;
    }

    ~StarpuSyncDefer()
    {
        --g_starpu_sync_defer_depth;
    }

    StarpuSyncDefer(StarpuSyncDefer const &) = delete;
    StarpuSyncDefer &operator=(StarpuSyncDefer const &) = delete;
};

inline void nnhaul_task_wait_for_all_unless_deferred()
{
    if (g_starpu_sync_defer_depth == 0)
    {
        starpu_task_wait_for_all_counted();
    }
}

inline void starpu_task_wait_for_all_unless_deferred()
{
    nnhaul_task_wait_for_all_unless_deferred();
}

} // namespace nntile
