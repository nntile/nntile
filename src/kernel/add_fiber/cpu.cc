/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cpu.cc
 * Per-element addition of a tensor and a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_fiber/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add_fiber
{

template<typename T>
void cpu(
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add a tensor and a broadcasted fiber with optional scaling on CPU
/*! Performs the following operation:
 * dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and src2[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] m: Size of the first mode of dst tensor.
 * @param[in] n: Size of the last mode of dst tensor.
 * @param[in] k: Size of the middle mode of dst and src2 tensors, and the first
 * mode of src1.
 * @param[in] batch: Size of the batch dimension.
 * @param[in] alpha: Scalar factor for src1.
 * @param[in] src1: Input contiguous vector with k*batch elements.
 * @param[in] beta: Scaling factor for src2.
 * @param[in] src2: Input contiguous tensor with m*k*n*batch elements.
 * @param[inout] dst: Output contiguous m-by-k-by-n-by-batch tensor.
 * */
{
    using Y = typename T::repr_t;
    const Y zero = 0.0, alpha_ = alpha, beta_ = beta;
    // Cycle over batch
    for(Index b = 0; b < batch; ++b)
    {
        // Cycle over input fiber src1
        for(Index i2 = 0; i2 < k; ++i2)
        {
            // Value to add to the output slice
            const Y src1_val = alpha_ * static_cast<Y>(src1[i2+b*k]);
            // Cycle over the third axis of output buffer dst
            for(Index i1 = 0; i1 < n; ++i1)
            {
                const T *src2_fiber = src2 + ((i1+b*n)*k+i2)*m;
                // Output fiber to be updated
                T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Read value from the second source tensor
                    const Y src2_val = static_cast<Y>(src2_fiber[i0]);
                    // And update output
                    dst_fiber[i0] = static_cast<T>(beta_ * src2_val + src1_val);
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp32_t *src1, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp64_t *src1, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const bf16_t *src1, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp16_t *src1, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
