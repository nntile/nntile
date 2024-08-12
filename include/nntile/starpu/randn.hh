/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/randn.hh
 * Randn operation on StarPU buffer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile::starpu::randn
{

// Randn operation on StarPU buffers
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

// Randn operation on StarPU buffers
template<typename T>
void cpu_ndim0(void *buffers[], void *cl_args)
    noexcept;

extern Codelet codelet_fp32, codelet_fp64, codelet_bf16,
       codelet_fp32_ndim0, codelet_fp64_ndim0, codelet_bf16_ndim0;

extern Codelet codelet_fp32_fast_tf32, codelet_fp32_fast_tf32_ndim0;

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
constexpr Codelet *codelet<fp32_fast_tf32_t>()
{
    return &codelet_fp32_fast_tf32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

template<typename T>
constexpr Codelet *codelet_ndim0()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet_ndim0<fp32_t>()
{
    return &codelet_fp32_ndim0;
}

template<>
constexpr Codelet *codelet_ndim0<bf16_t>()
{
    return &codelet_bf16_ndim0;
}

template<>
constexpr Codelet *codelet_ndim0<fp32_fast_tf32_t>()
{
    return &codelet_fp32_fast_tf32_ndim0;
}

template<>
constexpr Codelet *codelet_ndim0<fp64_t>()
{
    return &codelet_fp64_ndim0;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean, Scalar stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, Handle data,
        Handle tmp_index);

} // namespace nntile::starpu::randn
