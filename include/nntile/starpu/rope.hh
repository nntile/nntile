/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/add_fiber.hh
 * StarPU wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-05-27
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{
namespace rope
{

//! Structure for arguments
template<typename T>
struct args_t
{
    Index m;
    Index k;
    Index l;
};

// StarPU wrapper for kernel::add_fiber::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

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
void submit(Index m, Index k, Index l, Handle sin, Handle cos,
    Handle src, Handle dst);

} // namespace rope
} // namespace starpu
} // namespace nntile