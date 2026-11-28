/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/sync_defer.hh
 * TLS defer flag for NNHaul; wait calls ``nnhaul::wait``.
 * @version 2.0.0
 */

#pragma once

#include <nnhaul/nnhaul.hh>

#include <atomic>
#include <cstdint>

namespace nntile
{

extern thread_local int g_sync_defer_depth;

extern std::atomic<std::uint64_t> g_wait_for_all_count;

inline void task_wait_for_all_counted()
{
    ++g_wait_for_all_count;
    ::nnhaul::wait();
}

struct SyncDefer
{
    SyncDefer()
    {
        ++g_sync_defer_depth;
    }

    ~SyncDefer()
    {
        --g_sync_defer_depth;
    }

    SyncDefer(SyncDefer const &) = delete;
    SyncDefer &operator=(SyncDefer const &) = delete;
};

inline void nnhaul_task_wait_for_all_unless_deferred()
{
    if (g_sync_defer_depth == 0)
    {
        task_wait_for_all_counted();
    }
}

inline void task_wait_for_all_unless_deferred()
{
    nnhaul_task_wait_for_all_unless_deferred();
}

} // namespace nntile
