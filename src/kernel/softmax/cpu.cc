/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/softmax/cpu.cc
 * Softmax operation for a buffer on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/softmax/cpu.hh"
#include <cmath>

namespace nntile::kernel::softmax
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *maxsumexp, const T *src, T alpha,
        T *dst)
    noexcept
//! Compute softmax on a buffer along middle axis
/*! 
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] maxsumexp: Maximums and sums of exponents of slices
 * @param[in] src: Contiguous input array
 * @param[in] alpha: Scalar multiplier for the output
 * @param[out] dst: Contiguous output array
 * */
{
    Index src_dst_offset = 0;
    constexpr T zero = 0.0;
    // Outer loop by the last mode of dst and sumnorm arrays
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Middle loop by the middle mode of dst array
        for(Index i1 = 0; i1 < k; ++i1)
        {
            Index maxsumexp_offset = 2 * m * i2;
            // Inner loop by the first mode of dst and sumnorm arrays
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T val = src[src_dst_offset];
                // Max and sum of exponents
                const T max = maxsumexp[maxsumexp_offset];
                const T sum = maxsumexp[maxsumexp_offset+1];
                // Update value
                if(not std::isinf(val))
                {
                    dst[src_dst_offset] = alpha * std::exp(val-max) / sum;
                }
                else
                {
                    dst[src_dst_offset] = zero;
                }
                // Update pointers
                ++src_dst_offset;
                maxsumexp_offset += 2;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *maxsumexp,
        const fp32_t *src, fp32_t alpha, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *maxsumexp,
        const fp64_t *src, fp64_t alpha, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::softmax

