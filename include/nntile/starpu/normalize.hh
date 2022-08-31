/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/normalize.hh
 * Normalize operation for StarPU buffer
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
namespace normalize
{

//! Structure for arguments
template<typename T>
struct args_t
{
    Index m;
    Index n;
    Index k;
    Index l;
    T eps;
};

// Apply normalize along middle axis of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply normalize along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern StarpuCodelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr StarpuCodelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr StarpuCodelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index m, Index n, Index k, Index l, T eps,
        starpu_data_handle_t gamma_beta, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace normalize
} // namespace starpu
} // namespace nntile

