/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/sync_defer.hh
 * Defer ``starpu_task_wait_for_all`` during Runtime::execute_range submit.
 *
 * @version 1.1.0
 * */

#pragma once

#include <atomic>
#include <cstdint>
#include <starpu.h>

namespace nntile
{

//! Nesting depth: when > 0, sync wrappers skip ``starpu_task_wait_for_all``.
//! Defined once in ``src/starpu/sync_defer.cc`` (not inline) so every TU in
//! libnntile shares the same TLS cell.
extern thread_local int g_starpu_sync_defer_depth;

//! Counts every ``starpu_task_wait_for_all`` that actually runs.
extern std::atomic<std::uint64_t> g_starpu_wait_for_all_count;

inline void starpu_task_wait_for_all_counted()
{
    ++g_starpu_wait_for_all_count;
    starpu_task_wait_for_all();
}

//! RAII: make blocking core::* wrappers submit-only while in scope.
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

//! Wait unless ``StarpuSyncDefer`` is active (incremental async submit).
inline void starpu_task_wait_for_all_unless_deferred()
{
    if(g_starpu_sync_defer_depth == 0)
    {
        starpu_task_wait_for_all_counted();
    }
}

} // namespace nntile
