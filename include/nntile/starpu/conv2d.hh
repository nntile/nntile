/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/conv2d.hh
 * StarPU wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/defs.h>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace conv2d
{

//! Structure for arguments
struct args_t
{
    Index offset_n;
    Index offset_m;
    Index batch;
    Index src_n;
    Index src_m;
    Index kernel_n;
    Index kernel_m;
    Index dst_n;
    Index dst_m;
};

// StarPU wrapper for kernel::conv2d::cpu<T>
template <typename T> void cpu(void *buffers[], void *cl_args) noexcept;

extern Codelet codelet_fp32, codelet_fp64;

template <typename T> constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template <> constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template <> constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template <typename T>
void submit(Index offset_n, Index offset_m, Index batch, Index src_n,
            Index src_m, Handle src, Index kernel_n, Index kernel_m,
            Handle kernel, Index dst_n, Index dst_m, Handle dst);

} // namespace conv2d
} // namespace starpu
} // namespace nntile
