/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/log_scalar.hh
 * StarPU wrapper to log scalar value
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <string>

namespace nntile::starpu::log_scalar
{

struct args_t
{
    std::string name;
};

template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

extern Codelet codelet_fp32, codelet_fp64, codelet_bf16,
    codelet_fp32_fast_tf32;

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

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(const std::string &name, Handle value);

} // namespace nntile::starpu::log_scalar
