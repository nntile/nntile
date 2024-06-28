/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu/cpu.cc
 * ReLU operation on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/relu/cpu.hh"
#include <cmath>

namespace nntile::kernel::relu
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace ReLU operation on CPU
/*! Does the following per-element operation:
 * ReLU(z) = max(z, 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply ReLU
 * */
{
    constexpr T zero = 0;;
    for(Index i = 0; i < nelems; ++i)
    {
        T z = data[i];
        data[i] = std::fmax(z, zero);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::relu
