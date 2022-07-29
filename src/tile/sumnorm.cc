/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sumnorm.cc
 * Sum and Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/sumnorm.hh"
#include "nntile/kernel/cpu/sumnorm.hh"
#include <cmath>

namespace nntile
{

starpu_perfmodel sumnorm_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_sumnorm_fp32",
};

starpu_perfmodel sumnorm_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_sumnorm_fp64",
};

StarpuCodelet sumnorm_codelet_fp32("nntile_sumnorm_fp32",
        &sumnorm_perfmodel_fp32,
        {sumnorm_starpu_cpu<fp32_t>},
        {}
        );

StarpuCodelet sumnorm_codelet_fp64("nntile_sumnorm_fp64",
        &sumnorm_perfmodel_fp64,
        {sumnorm_starpu_cpu<fp64_t>},
        {}
        );

// Update sum and Euclidian norm
template<typename T>
void sumnorm_work(const Tile<T> &src, const Tile<T> &sumnorm, Index axis)
{
    // Get sizes
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = sumnorm.nelems / 2;
        k = src.shape[0];
    }
    else if(axis == src.ndim-1)
    {
        m = sumnorm.nelems / 2;
        n = 1;
        k = src.shape[axis];
    }
    else
    {
        m = src.stride[axis];
        n = src.matrix_shape[axis+1][1];
        k = src.shape[axis];
    }
    // Insert task
    int ret = starpu_task_insert(sumnorm_codelet<T>(),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                Starpu::STARPU_RW_COMMUTE,
                static_cast<starpu_data_handle_t>(sumnorm),
                0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation
template
void sumnorm_work<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &sumnorm,
        Index axis);

template
void sumnorm_work<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &sumnorm,
        Index axis);

} // namespace nntile

