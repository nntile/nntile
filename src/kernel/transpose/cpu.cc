/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cpu.cc
 * Transpose operation on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/transpose/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::transpose
{

template<typename T>
void cpu(Index m, Index n, Scalar alpha_, const T* src, T* dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha_: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    for(Index i = 0; i < m; ++i)
    {
        for(Index j = 0; j < n; ++j)
        {
            dst[i*n+j] = static_cast<T>(alpha * Y{src[i+j*m]});
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Scalar alpha, const fp32_t* src,
        fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Scalar alpha, const fp64_t* src,
        fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Scalar alpha, const bf16_t* src,
        bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::tranpose
