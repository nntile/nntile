/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cpu.cc
 * Bias operation over slices from a fiber of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/kernel/add_fiber/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace add_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *dst)
    noexcept
//! Bias over slices along the first and the last axes from a fiber of a tensor
/*! For a provided m-by-k-by-n output tensor apply bias along the first and the
 * last axes with m and n elements respectively from a fiber with k elements.
 * Mnemonically, the following operations are performed:
 *      dst[i,k,j] = beta*dst[i,k,j] + alpha*src[k]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of src tensor and the only mode of dst
 *      tensor
 * @param[in] alpha: Scalar factor for the src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-k-by-n array, that accumulates
 *      bias over slices along the first and the last axes
 * */
{
    constexpr T zero = 0.0;
    // Cycle over input fiber src
    for(Index i2 = 0; i2 < k; ++i2)
    {
        // Value to add to the output slice
        const T src_val = alpha * src[i2];
        // Cycle over the third axis of output buffer dst
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Output fiber to be updated
            T *dst_fiber = dst + (i1*k+i2)*m;
            // Overwrite or update output depending on beta
            if(beta == zero)
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Set output value
                    dst_fiber[i0] = src_val;
                }
            }
            else
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Read value from the output
                    T &dst_val = dst_fiber[i0];
                    // And update it
                    dst_val = beta*dst_val + src_val;
                }
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

} // namespace add_fiber
} // namespace kernel
} // namespace nntile

