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
 * @version 1.1.0
 * */

#include "nntile/kernel/add/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add_scalar
{

template<typename T>
void cpu(Index num_elements, Scalar alpha_, Scalar beta_, T* dst_)
    noexcept
//! Add scalar to buffer buffers on CPU
/*! dst[i] = alpha + beta*dst[i], where alpha and beta are scalars
 *
 * @param[in] num_elements: Size of the src and dst tensors
 * @param[in] alpha_: Scalar bias for the dst tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add_scalar operation
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto dst = reinterpret_cast<Y *>(dst_);
    const Y alpha{alpha_}, beta{beta_};
    for(Index i = 0; i < num_elements; ++i)
    {
        dst[i] = alpha + beta*dst[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_elements, Scalar alpha, Scalar beta, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index num_elements, Scalar alpha, Scalar beta, fp64_t* dst)
    noexcept;

} // namespace nntile::kernel::add_scalar
