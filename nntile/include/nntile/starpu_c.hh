/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/starpu_c.hh
 * StarPU C API with correct linkage from C++ translation units.
 *
 * starpu_disk.h is included by starpu.h before its extern "C" block, so
 * including <starpu.h> from C++ without a wrapper can mangle disk symbols.
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <starpu.h>

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <nntile/core/execution_worker.hh>

#ifdef starpu_task_insert
#undef starpu_task_insert
#endif

#ifdef STARPU_USE_FXT
#define nntile_starpu_task_insert_impl(cl, preferred_worker_id, ...)         \
    ((preferred_worker_id) < 0                                                \
            ? ::starpu_task_insert((cl),                                     \
                  STARPU_TASK_FILE,                                          \
                  __FILE__,                                                  \
                  STARPU_TASK_LINE,                                          \
                  __LINE__,                                                  \
                  ##__VA_ARGS__)                                             \
            : ::starpu_task_insert((cl),                                     \
                  STARPU_TASK_FILE,                                          \
                  __FILE__,                                                  \
                  STARPU_TASK_LINE,                                          \
                  __LINE__,                                                  \
                  STARPU_EXECUTE_ON_WORKER,                                  \
                  (preferred_worker_id),                                     \
                  ##__VA_ARGS__))

#define starpu_task_insert(cl, ...)                                          \
    ({                                                                       \
        int const _nntile_pref_worker =                                      \
            ::nntile::sched::preferred_starpu_worker_id();                   \
        nntile_starpu_task_insert_impl((cl), _nntile_pref_worker,            \
            ##__VA_ARGS__);                                                  \
    })
#else
#define nntile_starpu_task_insert_impl(cl, preferred_worker_id, ...)         \
    ((preferred_worker_id) < 0                                                \
            ? ::starpu_task_insert((cl), ##__VA_ARGS__)                      \
            : ::starpu_task_insert((cl),                                     \
                  STARPU_EXECUTE_ON_WORKER,                                  \
                  (preferred_worker_id),                                     \
                  ##__VA_ARGS__))

#define starpu_task_insert(cl, ...)                                          \
    ({                                                                       \
        int const _nntile_pref_worker =                                      \
            ::nntile::sched::preferred_starpu_worker_id();                   \
        nntile_starpu_task_insert_impl((cl), _nntile_pref_worker,            \
            ##__VA_ARGS__);                                                  \
    })
#endif

#endif // __cplusplus
