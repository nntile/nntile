/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/copy.hh
 * Copy StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/starpu/config.hh>

namespace nntile::starpu::copy
{

// Copy StarPU buffers on CPU
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Copy StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet;

void init();

void restrict_where(uint32_t where);

void restore_where();

//! Insert task to copy buffer
void submit(Handle src, Handle dst);

} // namespace nntile:starpu::copy
