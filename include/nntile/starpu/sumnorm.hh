/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/sumnorm.hh
 * Sum and Euclidian norm for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{

//! Structure for arguments
struct sumnorm_args
{
    Index m;
    Index n;
    Index k;
};

// Sum and Euclidian norm along middle axis of StarPU buffer on CPU
template<typename T>
void sumnorm_cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Sum and Euclidian norm along middle axis of StarPU buffer on CUDA
template<typename T>
void sumnorm_cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern StarpuCodelet sumnorm_codelet_fp32, sumnorm_codelet_fp64;

void sumnorm_init();

void sumnorm_restrict_where(uint32_t where);

void sumnorm_restore_where();

template<typename T>
void sumnorm(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

