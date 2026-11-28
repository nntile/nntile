/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/core/execution_worker.hh
 * NNHaul worker helpers for static execution schedules.
 *
 * @version 2.0.0
 * */

#pragma once

#include <string>

namespace nntile::sched
{

int count_execution_workers();


bool tile_op_has_cuda_kernel(std::string const &tile_op_name);

//! Map schedule logical worker to the NNHaul worker id for ``execute()``.
int worker_id_for_scheduled_op(
    int logical_worker,
    std::string const &tile_op_name);

} // namespace nntile::sched
