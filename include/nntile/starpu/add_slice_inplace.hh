/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/add_slice_inplace.hh
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile::starpu::add_slice_inplace
{

//! Structure for arguments
struct args_t
{
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

// StarPU wrapper for kernel::add_slice_inplace::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// StarPU wrapper for kernel::add_slice_inplace::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16,
               codelet_fp32_fast_fp16, codelet_fp32_fast_bf16;

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
constexpr Codelet *codelet<bf16_t>()
{
    return &codelet_bf16;
}

template<>
constexpr Codelet *codelet<fp32_fast_fp16_t>()
{
    return &codelet_fp32_fast_fp16;
}

template<>
constexpr Codelet *codelet<fp32_fast_bf16_t>()
{
    return &codelet_fp32_fast_bf16;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

template<>
constexpr Codelet *codelet<fp32_fast_tf32_t>()
{
    return &codelet_fp32_fast_tf32;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index m, Index n, Index k, Scalar alpha, Handle src, Scalar beta,
            Handle dst);

} // namespace nntile::starpu::add_slice_inplace
