/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/axpy.hh
 * AXPY operation for StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
// This also includes all definitions
#include <nntile/starpu/config.hh>

namespace nntile::starpu::axpy
{

//! Structure for arguments
struct args2_t
{
    Index nelems;
    Scalar alpha;
};

#ifdef NNTILE_USE_CBLAS
template<typename T>
void cpu_tensor_alpha(void *buffers[], void *cl_args)
    noexcept;

template<typename T>
void cpu_scalar_alpha(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;

template<typename T>
void cuda2(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_tensor_alpha_fp64, codelet_scalar_alpha_fp64;
extern Codelet codelet_tensor_alpha_fp32, codelet_scalar_alpha_fp32, codelet_scalar_alpha_fp32_fast_tf32;

template<typename T>
constexpr Codelet *codelet_tensor_alpha()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet_tensor_alpha<fp32_t>()
{
    return &codelet_tensor_alpha_fp32;
}

template<>
constexpr Codelet *codelet_tensor_alpha<fp64_t>()
{
    return &codelet_tensor_alpha_fp64;
}

template<typename T>
constexpr Codelet *codelet_scalar_alpha()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet_scalar_alpha<fp32_t>()
{
    return &codelet_scalar_alpha_fp32;
}

template<>
constexpr Codelet *codelet_scalar_alpha<fp32_fast_tf32_t>()
{
    return &codelet_scalar_alpha_fp32_fast_tf32;
}

template<>
constexpr Codelet *codelet_scalar_alpha<fp64_t>()
{
    return &codelet_scalar_alpha_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Handle alpha, Index nelems, Handle src, Handle dst);

template<typename T>
void submit(Scalar alpha, Index nelems, Handle src, Handle dst);

} // namespace nntile::starpu::axpy
