/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/rope3.hh
 * ROtary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-05-22
 * */

#pragma once

#include <nntile/kernel/rope3/cpu.hh>
#include <nntile/defs.h>
// #ifdef NNTILE_USE_CUDA
// #include <nntile/kernel/rope3/cuda.hh>
// #endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::rope3
/*! Low-level implementations of rotary positional embedding operation
 * */
namespace rope3
{

} // namespace rope3
} // namespace kernel
} // namespace nntile