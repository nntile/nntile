/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/data_access.hh
 * Access-mode names used by NNHaul call sites.
 * @version 2.0.0
 */

#pragma once


#include <nnhaul/nnhaul.hh>

//! Integer modes used by ``Tile::acquire`` and core submit call sites.
enum data_access_mode
{
    ACCESS_R = 1,
    ACCESS_W = 2,
    ACCESS_RW = 3,
    ACCESS_SCRATCH = 8,
    ACCESS_COMMUTE = 0
};

//! Drain NNHaul through the counted wait (defined in
//! nnhaul/sync_defer.hh).
namespace nntile
{
//! nnhaul::wait() through the counted wait (sync_defer.hh).
void nnhaul_wait_for_all_drained();
}

inline void task_wait_for_all()
{
    ::nntile::nnhaul_wait_for_all_drained();
}
