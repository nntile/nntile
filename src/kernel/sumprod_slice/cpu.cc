/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_slice/cpu.cc
 * Sums over fibers into a slice of a product of buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/sumprod_slice/cpu.hh"

namespace nntile::kernel::sumprod_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src1, const T *src2,
        T beta, T *dst)
    noexcept
//! Sums over fibers into a slice of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding fibers along second axis with k
 * elements, resulting in m-by-n output array dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum_l(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis of per-element products of src1 and src2.
 * */
{
    const Index mk = m * k;
    constexpr T zero = 0;
    // Cycle over column of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get corresponding fibers of both sources
            const T *src1_fiber = src1 + i2*mk + i1;
            const T *src2_fiber = src2 + i2*mk + i1;
            // Init sum of product of the fibers
            T sum = zero, c = zero, y, t;
            // Output value
            T &result = dst[i2*m+i1];
            // Cycle over fibers of inputs
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Update sum
                //sum += src1_fiber[i0*m] * src2_fiber[i0*m];
                y = src1_fiber[i0*m]*src2_fiber[i0*m] - c;
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
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src1,
        const fp32_t *src2, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src1,
        const fp64_t *src2, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::sumprod_slice
