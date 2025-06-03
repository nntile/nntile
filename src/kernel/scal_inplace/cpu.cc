/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scal_inplace/cpu.cc
 * Scal inplace operation on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scal_inplace/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::scal_inplace
{

template<typename T>
void cpu(Index nelems, Scalar alpha_, T* data)
    noexcept
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      data[i] = alpha * data[i]
 *
 * @param[in] nelems: Size of the data tensor
 * @param[in] alpha_: Scalar multiplier for the data tensor
 * @param[inout] data: Destination of the scal inplace operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = static_cast<T>(alpha * static_cast<Y>(data[i]));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, fp32_t* data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, fp64_t* data)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, bf16_t* data)
    noexcept;

} // namespace nntile::kernel::scal_inplace
