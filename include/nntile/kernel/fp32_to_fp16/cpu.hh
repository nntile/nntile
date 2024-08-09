/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp32_to_fp16/cpu.hh
 * Convert fp32_t array into fp16_t array on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::fp32_to_fp16
{

void cpu(Index nelems, const fp32_t *src, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::fp32_to_fp16
