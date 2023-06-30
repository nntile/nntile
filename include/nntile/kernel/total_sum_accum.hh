/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum.hh
 * Low-level kernels to compute total_sum_accum
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-16
 * */

#pragma once

#include <nntile/kernel/total_sum_accum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/total_sum_accum/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::total_sum_accum
/*! Low-level implementations of computing accumulated total sum 
 * */
namespace total_sum_accum
{

} // namespace total_sum_accum
} // namespace kernel
} // namespace nntile

