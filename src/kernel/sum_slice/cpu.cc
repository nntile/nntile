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
 * @version 1.1.0
 * */

#include "nntile/kernel/sum_slice/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sum_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src, Scalar beta_, T *dst)
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
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src_: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst_: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};
    const Index mk = m * k;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init sum over the fiber
            Y sum = zero, c = zero, y, t;
            // Output value
            T& result = dst[i2*m+i1];
            // Cycle over fiber elements and accumulate the sum
            for(Index i0 = 0; i0 < k; ++i0)
            {
                //sum += src_fiber[i0*m];
                y = Y{src_fiber[i0*m]} - c;
                t = sum + y;
                c = (t-sum) - y;
                sum = t;
            }
            // Update output value
            if(beta == zero)
            {
                result = static_cast<T>(alpha * sum);
            }
            else
            {
                result = static_cast<T>((beta * Y{result} - alpha*c) + alpha*sum);
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src,
        Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src,
        Scalar beta, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src,
        Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::sum_slice
