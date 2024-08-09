/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/addcdiv/cpu.hh
 * Per-element addcdiv operation for buffers on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::addcdiv
{

// Per-element addcdiv operation x = x + val * nom / (denom + eps)
template<typename T>
void cpu(Scalar val, Scalar eps, Index nelems, const T *nom, const T *denom, T *res)
    noexcept;

} // namespace nntile::kernel::addcdiv
