/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_slice/cpu.cc
 * Multiply a tensor per-element by a broadcasted slice on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/kernel/prod_slice/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace prod_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept
//! Product of a tensor and a broadcasted slice on CPU
/*! Multiply each element of an m-by-k-by-n tensor by a corresponding element
 * of another m-by-n tensor:
 *      dst[i,l,j] = dst[i,l,j] * src[i,j]
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
                // Read value from destination
                T &dst_val = dst_slice[i0*m];
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

} // namespace prod_slice
} // namespace kernel
} // namespace nntile

