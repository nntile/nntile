/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cpu.cc
 * Fill operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fill/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::fill
{

template<typename T>
void cpu(Index nelems, Scalar val_, T *data)
    noexcept
//! Fill operation on CPU
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val_: Input value
 * @params[out] data: Output buffer
 * */
{
    using Y = typename T::repr_t;
    const Y val{val_};
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = static_cast<T>(val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar val, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar val, fp64_t *data)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar val, bf16_t *data)
    noexcept;
} // namespace nntile::kernel::fill
