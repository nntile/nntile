/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/task_insert.hh
 * NNTile task submission for the StarPU subsystem.
 *
 * StarPU ops submit through nntile_starpu_task_insert(codelet, hint, ...)
 * instead of starpu_task_insert(). ``starpu_worker_hint`` is the logical
 * execution hint: pass -1 for no pinning, otherwise a StarPU worker id for
 * STARPU_EXECUTE_ON_WORKER.
 */

#pragma once

#include <starpu.h>

//! Submit a StarPU task, optionally pinned to ``starpu_worker_hint``.
#define nntile_starpu_task_insert(codelet, starpu_worker_hint, ...)           \
    (((starpu_worker_hint) < 0)                                                \
            ? ::starpu_task_insert((codelet), ##__VA_ARGS__)                   \
            : ::starpu_task_insert((codelet), STARPU_EXECUTE_ON_WORKER,        \
                  (starpu_worker_hint), ##__VA_ARGS__))
