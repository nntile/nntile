/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_slice/cpu.cc
 * Sums over fibers into a slice of a buffer on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/sum_slice/cpu.hh"
#include <cmath>

namespace nntile::kernel::sum_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *dst)
    noexcept
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis
 * */
{
    const Index mk = m * k;
    constexpr T zero = 0;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init sum over the fiber
            T sum = zero, c = zero, y, t;
            // Output value
            T &result = dst[i2*m+i1];
            // Cycle over fiber elements and accumulate the sum
            for(Index i0 = 0; i0 < k; ++i0)
            {
                //sum += src_fiber[i0*m];
                y = src_fiber[i0*m] - c;
                t = sum + y;
                c = (t-sum) - y;
                sum = t;
            }
            // Update output value
            if(beta == zero)
            {
                result = alpha * sum;
            }
            else
            {
                result = (beta*result-alpha*c) + alpha*sum;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src,
        fp32_t beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src,
        fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::sum_slice

