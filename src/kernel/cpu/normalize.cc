/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/normalize.cc
 * Normalize operation for a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/normalize.hh"
#include "nntile/kernel/args/normalize.hh"
#include "nntile/starpu.hh"
#include <cmath>

namespace nntile
{

//! Renormalize buffer along middle axis
//
// Provided m-by-k-by-n output tensor dst is renormalized along second axis
// with k elements. The following operations is applied:
//      dst[i, l, j] := (dst[i, l, j]-sumnorm[0, i, j]) / sqrt(sumnorm[1, i,
//      j]**2+eps) * gamma + beta
//
// @param[in] m: Size of the first mode of src and sumnorm tensors
// @param[in] n: Size of the last mode of src and sumnorm tensors
// @param[in] k: Size of the middle mode of src tensor
// @param[in] src: Input tensor to compute sums and norms of slices
// @param[inout] sumnorm: Sums and norms of slices
//
// @sa clear_kernel_cpu
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

//! Renormalize buffer along middle axis of StarPU buffer
//
// See normalize_kernel_cpu function for more info.
template<typename T>
void normalize_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<normalize_starpu_args<T> *>(cl_args);
    // Get interfaces
    auto interface = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    const T *gamma_beta = interface[0]->get_ptr<T>();
    T gamma = gamma_beta[0], beta = gamma_beta[1];
    const T *sumnorm = interface[1]->get_ptr<T>();
    T *dst = interface[2]->get_ptr<T>();
    normalize_kernel_cpu<T>(args->m, args->n, args->k, args->l, args->eps,
            gamma, beta, sumnorm, dst);
}

// Explicit instantiation of templates
template
void normalize_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void normalize_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

