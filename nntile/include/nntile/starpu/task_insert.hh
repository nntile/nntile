/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/task_insert.hh
 * NNTile task submission for the StarPU subsystem.
 *
 * StarPU ops must submit work through nntile_starpu_task_insert() instead of
 * starpu_task_insert() so static execution schedules can pin workers via
 * sched::preferred_starpu_worker_id() without redefining StarPU macros.
 */

#pragma once

#include <nntile/core/execution_worker.hh>
#include <nntile/starpu_c.hh>

//! Submit a StarPU task; pin to preferred worker when execution schedule is active.
/*!
 * When sched::preferred_starpu_worker_id() is non-negative, inserts
 * STARPU_EXECUTE_ON_WORKER before the remaining arguments (must precede the
 * trailing 0 sentinel). Otherwise forwards to starpu_task_insert unchanged.
 */
#define nntile_starpu_task_insert(codelet, ...)                                \
    ({                                                                         \
        int const _nntile_pref_worker =                                        \
            ::nntile::sched::preferred_starpu_worker_id();                     \
        (_nntile_pref_worker < 0)                                              \
                ? ::starpu_task_insert((codelet), ##__VA_ARGS__)               \
                : ::starpu_task_insert((codelet),                              \
                      STARPU_EXECUTE_ON_WORKER, _nntile_pref_worker,           \
                      ##__VA_ARGS__);                                          \
    })
