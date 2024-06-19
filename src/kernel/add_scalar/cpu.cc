/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_scalar/cpu.cc
 * Add scalar operation on buffer on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/add/cpu.hh"

namespace nntile::kernel::add_scalar
{

template<typename T>
void cpu(Index num_elements, T alpha, T beta, T* dst)
    noexcept
//! Add scalar to buffer buffers on CPU
/*! dst[i] = alpha + beta*dst[i], where alpha and beta are scalars
 *
 * @param[in] num_elements: Size of the src and dst tensors
 * @param[in] alpha: Scalar bias for the dst tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add_scalar operation
 * */
{
    for (Index i = 0; i < num_elements; ++i)
    {
        dst[i] = alpha + beta * dst[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_elements, fp32_t alpha, fp32_t beta, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index num_elements, fp64_t alpha, fp64_t beta, fp64_t* dst)
    noexcept;

} // namespace nntile::kernel::add_scalar

