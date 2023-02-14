/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/addcdiv/cpu.hh
 * Per-element maximum of two buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace addcdiv
{

// Per-element addcdiv operation x = x + val * nom / (denom + eps)
template<typename T>
void cpu(T val, T eps, Index nelems, const T *nom, const T *denom, T *res)
    noexcept;

} // namespace addcdiv
} // namespace kernel
} // namespace nntile