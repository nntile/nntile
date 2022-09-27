/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/clear.hh
 * Clear a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-27
 * */

#pragma once

#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace clear
{

// Clear a StarPU buffer on CPU
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply bias along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet;

void init();

void restrict_where(uint32_t where);

void restore_where();

//! Insert task to clear buffer
void submit(Handle data);

} // namespace clear
} // namespace starpu
} // namespace nntile

