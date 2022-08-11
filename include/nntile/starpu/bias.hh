/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/bias.hh
 * Bias operation on a StarPU buffer
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

//! Structure for arguments
struct bias_args
{
    Index m;
    Index n;
    Index k;
};

// Apply bias along middle axis of StarPU buffer on CPU
template<typename T>
void bias_cpu(void *buffers[], void *cl_args)
    noexcept;

extern StarpuCodelet bias_codelet_fp32, bias_codelet_fp64;

void bias_restrict_where(uint32_t where);

void bias_restore_where();

template<typename T>
void bias(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

