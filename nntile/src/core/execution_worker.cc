#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/core/execution_worker.cc
 *
 * @version 1.1.0
 * */

#include "nntile/core/execution_worker.hh"

#include <starpu.h>

namespace nntile::sched
{

int count_execution_workers()
{
    if (!starpu_is_initialized())
    {
        return 1;
    }
    int const ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    if (ncuda > 0)
    {
        return ncuda;
    }
    int const ncpu = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    return ncpu > 0 ? ncpu : 1;
}

int logical_worker_to_starpu_id(int logical_worker, bool use_cuda_workers)
{
    if (!starpu_is_initialized() || logical_worker < 0)
    {
        return -1;
    }
    if (use_cuda_workers)
    {
        int const ncuda =
            starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
        if (ncuda <= 0)
        {
            return -1;
        }
        return starpu_worker_get_by_type(
            STARPU_CUDA_WORKER, logical_worker % ncuda);
    }
    int const ncpu = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    if (ncpu <= 0)
    {
        return -1;
    }
    return starpu_worker_get_by_type(
        STARPU_CPU_WORKER, logical_worker % ncpu);
}

bool tile_op_requires_cpu_worker(std::string const &tile_op_name)
{
    return tile_op_name == "TILE_LOG_SCALAR" ||
        tile_op_name == "TILE_RANDN";
}

int starpu_worker_id_for_scheduled_op(
    int logical_worker,
    bool use_cuda_workers,
    std::string const &tile_op_name)
{
    if (tile_op_requires_cpu_worker(tile_op_name))
    {
        return logical_worker_to_starpu_id(logical_worker, false);
    }
    return logical_worker_to_starpu_id(logical_worker, use_cuda_workers);
}

} // namespace nntile::sched
