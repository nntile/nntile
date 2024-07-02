/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/rope.hh
 * Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-05-22
 * */

#pragma once

#include <nntile/kernel/rope/cpu.hh>
#include <nntile/defs.h>
// #ifdef NNTILE_USE_CUDA
// #include <nntile/kernel/embedding/cuda.hh>
// #endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::rope
/*! Low-level implementations of rotary positional embedding operation
 * */
namespace rope
{

} // namespace rope
} // namespace kernel
} // namespace nntile

