/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cpu.cc
 * Bias operation over fibers from a slice of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/kernel/add_slice/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace add_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *dst)
    noexcept
//! Bias over fibers along middle axis from a slice of a tensor
/*! For a provided m-by-k-by-n output tensor apply bias along second axis
 * with k elements from m-by-n input tensor.
 * Mnemonically, the following operations are performed:
 *      dst[i,k,j] = beta*dst[i,k,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for the src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-k-by-n array, that accumulates
 *      bias over fibers along middle axis
 * */
{
    const Index mk = m * k;
    constexpr T zero = 0.0;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the output array dst
            T *dst_fiber = dst + i2*mk + i1;
            // Value to add to the output fiber
            const T src_val = alpha * src[i2*m+i1];
            // Overwrite or update output depending on beta
            if(beta == zero)
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Set output value
                    dst_fiber[i0*m] = src_val;
                }
            }
            else
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Read value from the output
                    T &dst_val = dst_fiber[i0*m];
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

} // namespace add_slice
} // namespace kernel
} // namespace nntile

