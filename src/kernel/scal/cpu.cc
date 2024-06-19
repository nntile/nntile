/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scal/cpu.cc
 * Scal operation on buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/scal/cpu.hh"

namespace nntile::kernel::scal
{

template<typename T>
void cpu(Index nelems, T alpha, const T* src, T* dst)
    noexcept
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the scal operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = alpha * src[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t alpha, const fp32_t* src, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t alpha, const fp64_t* src, fp64_t* dst)
    noexcept;

} // namespace nntile::kernel::scal

