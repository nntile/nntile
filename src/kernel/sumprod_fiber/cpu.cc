/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_fiber/cpu.cc
 * Sums over slices into a fiber of a product of buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sumprod_fiber/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sumprod_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src1, const T *src2,
        Scalar beta_, T *dst)
    noexcept
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha_: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};
    // Cycle over output vector
    for(Index i2 = 0; i2 < k; ++i2)
    {
        // Init sum of product of the slices
        Y sum = zero, c = zero, y, t;
        // Output value
        T &result = dst[i2];
        // Cycle over column of src1 and src2
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Get corresponding fibers of both sources
            const T *src1_fiber = src1 + (i1*k+i2)*m;
            const T *src2_fiber = src2 + (i1*k+i2)*m;
            // Cycle over fibers of inputs
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Update sum
                sum += Y{src1_fiber[i0]} * Y{src2_fiber[i0]};
                //y = src1_fiber[i0*m]*src2_fiber[i0*m] - c;
                //t = sum + y;
                //c = (t-sum) - y;
                //sum = t;
            }
        }
        // Update output value
        if(beta == zero)
        {
            result = static_cast<T>(alpha * sum);
        }
        else
        {
            result = static_cast<T>((beta * Y{result} - alpha * c) + alpha * sum);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src1,
        const fp32_t *src2, Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src1,
        const fp64_t *src2, Scalar beta, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src1,
        const bf16_t *src2, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::sumprod_fiber
