/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/normalize.cc
 * Normalize operation for buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/normalize.hh"
#include <starpu_data_interfaces.h>
#include <cmath>

namespace nntile
{

template<typename T>
void normalize_kernel_cpu(Index m, Index n, Index k, Index l, T eps, T gamma,
        T beta, const T *sumnorm, T *dst)
    noexcept
{
    Index dst_offset = 0;
    constexpr T one = 1;
    const T invl = one / T(l);
    const T rinvl = std::sqrt(invl);
    const T reps = std::sqrt(eps);
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
                const T sum = sumnorm[src_offset];
                const T mean = sum * invl;
                const T norm = sumnorm[src_offset+1];
                const T rms = norm * rinvl;
                T dev;
                if(rms > reps)
                {
                    T tmp = mean/rms, tmp2 = reps/rms;
                    T ssq = one - tmp*tmp;
                    ssq += tmp2*tmp2;
                    dev = rms * std::sqrt(ssq);
                }
                else
                {
                    T tmp = rms/reps, tmp2 = mean/reps;
                    T ssq = tmp*tmp - tmp2*tmp2;
                    ssq += one;
                    dev = reps * std::sqrt(ssq);
                }
                // Normalization
                val = (val-mean) / dev;
                // Renormalization
                val = val*gamma + beta;
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

template<typename T>
void normalize_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Source (avg_dev) is a 2-by-m-by-n tile, which contains mean and
    // deviation values
    // Destination is an m-by-k-by-n tile
    // Both source and destination are Fortran-contiguous
    Index m, n, k, l;
    T eps;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k, &l, &eps);
    const T *gamma_beta = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(
                buffers[0]));
    const T *sumnorm = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    T gamma = gamma_beta[0], beta = gamma_beta[1];
    T *dst = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[2]));
    normalize_kernel_cpu<T>(m, n, k, l, eps, gamma, beta, sumnorm, dst);
}

// Explicit instantiation of templates
template
void normalize_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void normalize_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

