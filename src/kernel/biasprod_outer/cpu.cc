/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/biasprod_outer/cpu.cc
 * Bias-like product along outer axes operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/kernel/biasprod_outer/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace biasprod_outer
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept
//! Bias-like product along outer axes on CPU
/*! For a provided m-by-k-by-n output tensor dst apply bias-like product along
 * the first and the third axes from an input tensor src of shape (k):
 *      dst[:, i, :] *= src[i]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst and the only mode of src
 *      tensors
 * @param[in] src: Source of the bias
 * @param[inout] dst: Destination of the bias
 * */
{
    // Cycle over input buffer
    for(Index i2 = 0; i2 < k; ++i2)
    {
        const T src_val = src[i2];
        // Cycle over the third axis of output buffer
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Output slice to be updated
            T *dst_slice = dst + (i1*k+i2)*m;
            // Cycle over the first axis of output buffer
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0];
                // And update it
                dst_val = dst_val * src_val;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace biasprod_outer
} // namespace kernel
} // namespace nntile

