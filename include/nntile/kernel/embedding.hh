/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding.hh
 * Embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/embedding/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/embedding/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::embedding
/*! Low-level implementations of embedding operation
 * */
namespace nntile::kernel::embedding
{

} // namespace nntile::kernel::embedding
