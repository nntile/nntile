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
 * @date 2022-08-05
 * */

#pragma once

#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{

// Clear a StarPU buffer on CPU
void clear_cpu(void *buffers[], void *cl_args)
    noexcept;

extern starpu_perfmodel clear_perfmodel;

extern StarpuCodelet clear_codelet;

void clear_restrict_where(uint32_t where);

void clear_restore_where();

//! Insert task to clear buffer
void clear(starpu_data_handle_t src);

} // namespace starpu
} // namespace nntile

