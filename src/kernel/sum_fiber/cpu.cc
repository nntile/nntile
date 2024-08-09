/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_fiber/cpu.cc
 * Sums over slices into a fiber of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum_fiber/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sum_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
    noexcept
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k,b] = beta*dst[k,b] + alpha*sum(src[:,k,:,b])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};
    // Cycle over batch
    for(Index b = 0; b < batch; ++b)
    {
        // Cycle over the only axis of output buffer
        for(Index i2 = 0; i2 < k; ++i2)
        {
            // Init sum
            Y sum = zero, c = zero, y, t;
            // Cycle over the third axis of input buffer
            for(Index i1 = 0; i1 < n; ++i1)
            {
                // Get sum of a corresponding slice
                const T *src_slice = src + ((i1+b*n)*k+i2)*m;
                // Cycle over the first axis of input buffer
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Read value from source
                    Y val = Y{src_slice[i0]};
                    // Update sum
                    //sum += val;
                    y = val - c;
                    t = sum + y;
                    c = (t-sum) - y;
                    sum = t;
                }
            }
            // Save result
            if(beta == zero)
            {
                sum *= alpha;
            }
            else
            {
                sum = (beta * Y{dst[i2+b*k]} - alpha * c) + alpha * sum;
            }
            dst[i2+b*k] = static_cast<T>(sum);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::sum_fiber
