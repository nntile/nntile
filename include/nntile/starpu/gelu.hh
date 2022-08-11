/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/gelu.hh
 * GeLU operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-11
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{

// Apply gelu along middle axis of StarPU buffer on CPU
template<typename T>
void gelu_cpu(void *buffers[], void *cl_args)
    noexcept;

extern StarpuCodelet gelu_codelet_fp32, gelu_codelet_fp64;

void gelu_restrict_where(uint32_t where);

void gelu_restore_where();

template<typename T>
void gelu(Index nelems, starpu_data_handle_t data);

} // namespace starpu
} // namespace nntile

