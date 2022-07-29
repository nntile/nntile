/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/bias2.cc
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/bias.hh"
#include <starpu_data_interfaces.h>

namespace nntile
{

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
    T *avg_dev = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    bias2_kernel_cpu<T>(m, n, k, avg_dev, dst);
}

// Explicit instantiation of templates
template
void bias2_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void bias2_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

