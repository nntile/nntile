/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/fp32_to_fp16.hh
 * Convert fp32_t array into fp16_t array on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#pragma once

//#include <nntile/kernel/fp32_to_fp16/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/fp32_to_fp16/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::fp32_to_fp16
/*! Low-level implementations of forward ReLU operation
 * */
namespace fp32_to_fp16
{

} // namespace fp32_to_fp16
} // namespace kernel
} // namespace nntile

