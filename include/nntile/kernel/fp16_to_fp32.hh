/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp16_to_fp32.hh
 * Convert fp16_t array into fp32_t array on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/fp16_to_fp32/cpu.hh>
#include <nntile/kernel/fp16_to_fp32/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::fp16_to_fp32
/*! Low-level implementations of convertion fp16 to fp32 operation
 * */
namespace nntile::kernel::fp16_to_fp32
{

} // namespace nntile::kernel::fp16_to_fp32
