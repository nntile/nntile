/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp32_to_fp16/cpu.hh
 * Convert fp32_t array into fp16_t array on CPU
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
namespace fp32_to_fp16
{

void cpu(Index nelems, const fp32_t *src, fp16_t *dst)
    noexcept;

} // namespace fp32_to_fp16
} // namespace kernel
} // namespace nntile

