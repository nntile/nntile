/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp16_to_fp32/cpu.hh
 * Convert fp16_t array into fp32_t array on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-09
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace fp16_to_fp32
{

void cpu(Index nelems, const fp16_t *src, fp32_t *dst)
    noexcept;

} // namespace fp16_to_fp32
} // namespace kernel
} // namespace nntile

