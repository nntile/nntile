/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/bias/cpu.cc
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#include "nntile/kernel/bias/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace bias
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept
//! Bias along middle axis on CPU
/*! For a provided m-by-k-by-n output tensor dst apply bias along second axis
 * with k elements from m-by-n input tensor src. A value src[i, j] is added to
 * an entire slice dst[i, :, j].
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] src: Source of the bias
 * @param[inout] dst: Destination of the bias
 * */
{
    Index src_offset = 0;
    const Index mk = m * k;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Output slice to be updated
            T *dst_slice = dst + i2*mk + i1;
            const T src_val = src[src_offset];
            ++src_offset;
            // Cycle over slice of output buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0*m];
                // And update it
                dst_val = dst_val + src_val;
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

template<typename T>
void cpu(T val, Index num_elements, T *y)
    noexcept
{
    // Do nothing fancy if x is zero
    if(val == T(0))
    {
        return;
    }
    for (Index i = 0; i < num_elements; ++i)
        y[i] += val;
}

// Explicit instantiation
template
void cpu<fp32_t>(fp32_t val, Index num_elements, fp32_t *y)
    noexcept;

template
void cpu<fp64_t>(fp64_t val, Index num_elements, fp64_t *y)
    noexcept;

} // namespace bias
} // namespace kernel
} // namespace nntile

