/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/bias.cc
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
    T *src = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    bias_kernel_cpu<T>(m, n, k, src, dst);
}

// Explicit instantiation of templates
template
void bias_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void bias_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

