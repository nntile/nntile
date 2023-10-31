/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/copy.hh
 * Copy StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-18
 * */

#pragma once

#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace copy
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

} // namespace copy
} // namespace starpu
} // namespace nntile


