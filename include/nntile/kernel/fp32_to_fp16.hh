/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp32_to_fp16.hh
 * Convert fp32_t array into fp16_t array on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/fp32_to_fp16/cpu.hh>
#include <nntile/kernel/fp32_to_fp16/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::fp32_to_fp16
/*! Low-level implementations of convertion fp32 to fp16 operation
 * */
namespace nntile::kernel::fp32_to_fp16
{

} // namespace nntile::kernel::fp32_to_fp16
