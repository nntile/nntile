/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/bias.cc
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/bias.hh"
#include "nntile/kernel/cpu/bias.hh"

#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/bias.hh"
#endif // NNTILE_USE_CUDA

namespace nntile
{

struct starpu_perfmodel bias_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias_fp32",
};

struct starpu_perfmodel bias_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias_fp64",
};

StarpuCodelet bias_codelet_fp32("nntile_bias_fp32",
        &bias_perfmodel_fp32,
        {bias_starpu_cpu<fp32_t>},
#       ifdef NNTILE_USE_CUDA
            {bias_starpu_cuda<fp32_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

StarpuCodelet bias_codelet_fp64("nntile_bias_fp64",
        &bias_perfmodel_fp64,
        {bias_starpu_cpu<fp64_t>},
#       ifdef NNTILE_USE_CUDA
            {bias_starpu_cuda<fp64_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

void bias_restrict_where(uint32_t where)
{
    bias_codelet_fp32.restrict_where(where);
    bias_codelet_fp64.restrict_where(where);
}

void bias_restore_where()
{
    bias_codelet_fp32.restore_where();
    bias_codelet_fp64.restore_where();
}

// Bias operation over single axis
template<typename T>
void bias_work(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = src.nelems;
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = src.nelems;
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
    int ret = starpu_task_insert(bias_codelet<T>(),
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            Starpu::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(dst.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation of template
template
void bias_work(const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

// Explicit instantiation of template
template
void bias_work(const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

} // namespace nntile

