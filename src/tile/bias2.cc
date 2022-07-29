/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/bias2.cc
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/bias2.hh"
#include "nntile/kernel/cpu/bias2.hh"

#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/bias2.hh"
#endif // NNTILE_USE_CUDA

namespace nntile
{

starpu_perfmodel bias2_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias2_fp32",
};

starpu_perfmodel bias2_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias2_fp64",
};

StarpuCodelet bias2_codelet_fp32("nntile_bias2_fp32",
        &bias2_perfmodel_fp32,
        {bias2_starpu_cpu<fp32_t>},
#       ifdef NNTILE_USE_CUDA
            {bias2_starpu_cuda<fp32_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

StarpuCodelet bias2_codelet_fp64("nntile_bias2_fp64",
        &bias2_perfmodel_fp64,
        {bias2_starpu_cpu<fp64_t>},
#       ifdef NNTILE_USE_CUDA
            {bias2_starpu_cuda<fp64_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

void bias2_restrict_where(uint32_t where)
{
    bias2_codelet_fp32.restrict_where(where);
    bias2_codelet_fp64.restrict_where(where);
}

void bias2_restore_where()
{
    bias2_codelet_fp32.restore_where();
    bias2_codelet_fp64.restore_where();
}

// Normalization operation over single axis
template<typename T>
void bias2_work(const Tile<T> &avg_dev, const Tile<T> &dst, Index axis)
{
    // Reshape inputs for simplicity: src -> (2,m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = avg_dev.nelems / 2; // 2 elements per single n
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = avg_dev.nelems / 2; // 2 elements per single m
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    int ret = starpu_task_insert(bias2_codelet<T>(),
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(avg_dev),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(dst.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation of template
template
void bias2_work(const Tile<fp32_t> &avg_dev, const Tile<fp32_t> &dst,
        Index axis);

// Explicit instantiation of template
template
void bias2_work(const Tile<fp64_t> &avg_dev, const Tile<fp64_t> &dst,
        Index axis);

} // namespace nntile

