/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/normalize/cpu.cc
 * Normalize operation for a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/normalize/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::normalize
{

template<typename T>
void cpu(Index m, Index n, Index k, Index size, Scalar eps_, const T *gamma_,
        const T *beta_, const T *sumnorm_, T *dst_)
    noexcept
//! Renormalize buffer along middle axis
/*! Provided m-by-k-by-n output array dst is renormalized along second axis
 * with k elements. The following operations is applied:
 *      dst[i, :, j] := (dst[i, :, j]-mean(i, j)) / sqrt(var(i, j)+eps)
 *          * gamma + beta
 * where mean and var functions are computed as follows:
 *      mean(i, j) = sumnorm[0, i, j] / size
 *      var(i, j) = sumnorm[1, i, j]^2/size - mean(i,j)^2
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] size: Number of elements used to calculate sum and Euclidean norm
 * @param[in] eps_: Regularization parameter for variance. eps > 0
 * @param[in] gamma_: Deviation for the renormalized output
 * @param[in] beta_: Mean value for the renormalized output
 * @param[in] sumnorm_: Sums and norms of slices
 * @param[in] dst_: Contiguous output array
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto gamma = reinterpret_cast<const Y *>(gamma_);
    auto beta = reinterpret_cast<const Y *>(beta_);
    auto sumnorm = reinterpret_cast<const Y *>(sumnorm_);
    auto dst = reinterpret_cast<Y *>(dst_);
    const Y eps{eps_};
    Index dst_offset = 0;
    constexpr Y one{1.0};
    const Y invl = one / Y(size);
    const Y rinvl = std::sqrt(invl);
    const Y reps = std::sqrt(eps);
    // Outer loop by the last mode of dst and sumnorm arrays
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Middle loop by the middle mode of dst array
        for(Index i1 = 0; i1 < k; ++i1)
        {
            Index src_offset = 2 * m * i2;
            // Inner loop by the first mode of dst and sumnorm arrays
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                Y &val = dst[dst_offset];
                // Corresponding mean and root-mean-square
                const Y sum = sumnorm[src_offset];
                const Y mean = sum * invl;
                const Y norm = sumnorm[src_offset+1];
                const Y rms = norm * rinvl;
                // Deviation=sqrt(rms*rms-mean*mean+reps*reps)
                Y dev;
                // Although in theory tmp<=1 it is not always true in practice
                // due presence of rounding errors
                Y tmp = std::fabs(mean) / rms;
                // Check if rounding errors broke theoretical invariant
                if(tmp >= one)
                {
                    dev = reps;
                }
                else if(rms > reps)
                {
                    Y ssq = one - tmp*tmp;
                    Y tmp2 = reps / rms;
                    ssq += tmp2*tmp2;
                    dev = rms * std::sqrt(ssq);
                }
                else
                {
                    Y ssq = one - tmp*tmp;
                    Y tmp2 = rms / reps;
                    ssq *= tmp2 * tmp2;
                    ssq += one;
                    dev = reps * std::sqrt(ssq);
                }
                // Normalization
                val = (val-mean) / dev;
                // Renormalization
                val = val*gamma[0] + beta[0];
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index size, Scalar eps,
        const fp32_t *gamma, const fp32_t *beta, const fp32_t *sumnorm,
        fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index size, Scalar eps,
        const fp64_t *gamma, const fp64_t *beta, const fp64_t *sumnorm,
        fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::normalize
