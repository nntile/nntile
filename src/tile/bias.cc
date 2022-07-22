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

namespace nntile
{

template<typename T>
void bias_kernel_cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept
{
    Index src_offset = 0;
    const Index mk = m * k;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            T *dst_slice = dst + i2*mk + i1;
            const T src_val = src[src_offset];
            ++src_offset;
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0*m];
                dst_val = dst_val + src_val;
            }
        }
    }
}

template<typename T>
void bias_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    bias_kernel_cpu<T>(m, n, k, src, dst);
}

starpu_perfmodel bias_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias_fp32",
};

starpu_perfmodel bias_perfmodel_fp64 =
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
    int ret = starpu_task_insert(bias_get_codelet<T>(),
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

