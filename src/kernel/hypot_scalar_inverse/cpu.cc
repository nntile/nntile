/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot_scalar_inverse/cpu.cc
 * Inverse of a hypot operation of a buffer and a scalar on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/hypot_scalar_inverse/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::hypot_scalar_inverse
{

template<typename T>
void cpu(Index nelems, Scalar eps_, Scalar alpha_, T* dst)
    noexcept
//! Inverse of a hypot of a buffer and a scalar on CPU
/*! Performs the following operation:
 *      dst[i] = 1.0 / hypot(alpha*dst[i], eps),
 * where alpha and eps are non-zero scalars.
 *
 * @param[in] nelems: Size of the dst tensor
 * @param[in] eps_: Scalar to be added to the hypot result
 * @param[in] alpha_: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot operation
 * */
{
    using Y = typename T::repr_t;
    const Y eps{eps_}, alpha{alpha_};
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = static_cast<T>(Y{1.0} / std::hypot(alpha*Y{dst[i]}, eps));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar eps, Scalar alpha, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar eps, Scalar alpha, fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar eps, Scalar alpha, bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::hypot_scalar_inverse
