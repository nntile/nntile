/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh/cpu.hh
 * Approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::gelutanh
{

// Approximate GeLU operation on a buffer on CPU
template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept;

} // namespace nntile::kernel::gelutanh
