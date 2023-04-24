/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/sum_slice.hh
 * Sum over fibers into a slice of a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev 
 * @author Konstantin Sozykin
 * @date 2023-04-24
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace sum_slice
{

//! Structure for arguments
template<typename T>
struct args_t
{
    Index m;
    Index n;
    Index k;
    T alpha;
    T beta;
};

// StarPU wrapper for kernel::sum_slice::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// StarPU wrapper for kernel::sum_slice::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index m, Index n, Index k, T alpha, Handle src, T beta,
        Handle sum_dst);

} // namespace sum_slice
} // namespace starpu
} // namespace nntile

