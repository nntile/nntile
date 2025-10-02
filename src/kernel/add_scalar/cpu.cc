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
void cpu(Index num_elements, Scalar alpha, Scalar beta, T* dst)
    noexcept
//! Add scalar to buffer on CPU
/*! Perform element-wise operation: dst[i] = alpha + beta * dst[i]
 *
 * This operation modifies the destination buffer in-place by adding a scalar
 * value (alpha) and scaling the existing values by a scalar factor (beta).
 *
 * @param[in] num_elements: Number of elements in the destination buffer
 * @param[in] alpha: Scalar value to add to each element
 * @param[in] beta: Scalar multiplier for each element before adding alpha
 * @param[inout] dst: Destination buffer to modify in-place
 * */
{
    using Y = typename T::repr_t;
    const Y alpha_val{alpha}, beta_val{beta};
    for(Index i = 0; i < num_elements; ++i)
    {
        dst[i] = T{alpha_val + beta_val * Y{dst[i]}};
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_elements, Scalar alpha, Scalar beta, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index num_elements, Scalar alpha, Scalar beta, fp64_t* dst)
    noexcept;

template
void cpu<fp16_t>(Index num_elements, Scalar alpha, Scalar beta, fp16_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index num_elements, Scalar alpha, Scalar beta, bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::add_scalar
