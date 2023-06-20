/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding.hh
 * Embeddings from vocabulary within buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#pragma once

#include <nntile/kernel/embedding/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/embedding/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::embedding
/*! Low-level implementations of embedding operation
 * */
namespace embedding
{

} // namespace embedding
} // namespace kernel
} // namespace nntile

