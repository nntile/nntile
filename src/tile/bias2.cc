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

namespace nntile
{

// CPU codelet for normalization over single axis
template<typename T>
void bias2_kernel_cpu(Index m, Index n, Index k, const T *avg_dev, T *dst)
    noexcept
{
    Index dst_offset = 0;
    // Outer loop by the last mode of source and destination tiles
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Middle loop by the middle mode of destination tile
        for(Index i1 = 0; i1 < k; ++i1)
        {
            Index src_offset = 2 * m * i2;
            // Inner loop by the first mode of source and destination tiles
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T &val = dst[dst_offset];
                // Corresponding mean and deviation
                const T &avg = avg_dev[src_offset];
                const T &dev = avg_dev[src_offset+1];
                // Normalization
                val = (val-avg) / dev;
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

template<typename T>
void bias2_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Source (avg_dev) is a 2-by-m-by-n tile, which contains mean and
    // deviation values
    // Destination is an m-by-k-by-n tile
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    T *avg_dev = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    bias2_kernel_cpu<T>(m, n, k, avg_dev, dst);
}

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
    int ret = starpu_task_insert(bias2_get_codelet<T>(),
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

