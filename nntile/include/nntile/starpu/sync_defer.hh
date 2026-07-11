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

#include <starpu.h>

namespace nntile
{

//! Nesting depth: when > 0, sync wrappers skip ``starpu_task_wait_for_all``.
inline thread_local int g_starpu_sync_defer_depth = 0;

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
        starpu_task_wait_for_all();
    }
}

} // namespace nntile
