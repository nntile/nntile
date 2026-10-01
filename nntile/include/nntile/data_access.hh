/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/data_access.hh
 * Access-mode names shared with StarPU call sites. NNHaul builds only.
 */

#pragma once

#ifdef NNTILE_USE_NNHAUL

#include <nnhaul/nnhaul.hh>

//! Integer modes used by ``Tile::acquire`` and core submit call sites.
enum starpu_data_access_mode
{
    STARPU_R = 1,
    STARPU_W = 2,
    STARPU_RW = 3,
    STARPU_REDUX = 4,
    STARPU_SCRATCH = 8,
    STARPU_COMMUTE = 0
};

//! Tests call the StarPU drain by this name. On this build it joins NNHaul.
inline void starpu_task_wait_for_all()
{
    ::nnhaul::wait();
}

#endif // NNTILE_USE_NNHAUL
