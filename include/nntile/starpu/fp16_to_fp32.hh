/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/fp16_to_fp32.hh
 * Convert fp16_t array into fp32_t array on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{
namespace fp16_to_fp32
{

//void cpu(void *buffers[], void *cl_args)
//    noexcept;

#ifdef NNTILE_USE_CUDA
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet;

void init();

void restrict_where(uint32_t where);

void restore_where();

void submit(Index nelems, Handle src, Handle dst);

} // namespace fp16_to_fp32
} // namespace starpu
} // namespace nntile

