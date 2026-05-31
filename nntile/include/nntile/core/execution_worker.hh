/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/core/execution_worker.hh
 * StarPU worker helpers for static execution schedules.
 *
 * @version 1.1.0
 * */

#pragma once

namespace nntile::sched
{

int count_execution_workers();

int logical_worker_to_starpu_id(int logical_worker, bool use_cuda_workers);

} // namespace nntile::sched
