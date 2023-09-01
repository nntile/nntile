/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cpu.cc
 * Transpose operation on buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-20
 * */

#include "nntile/kernel/transpose/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace transpose
{

template<typename T>
void cpu(Index m, Index n, T alpha, const T* src, T* dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    for(Index i = 0; i < m; ++i)
    {
        for(Index j = 0; j < n; ++j)
        {
            dst[i*n+j] = alpha * src[i+j*m];
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, fp32_t alpha, const fp32_t* src,
        fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, fp64_t alpha, const fp64_t* src,
        fp64_t* dst)
    noexcept;

} // namespace tranpose
} // namespace kernel
} // namespace nntile

