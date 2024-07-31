/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/conv2d_v2_bwd_input_inplace.hh
 * Backward of 2D-Convolution for gradient over input
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/defs.h>
#include <nntile/starpu/config.hh>

namespace nntile::starpu::conv2d_v2_bwd_input_inplace
{

//! Structure for arguments
struct args_t
{
    Index src_m;
    Index src_n;
    Index in_channels;
    Index batch;
    Index offset_m;
    Index offset_n;
    Scalar alpha;
    Index kernel_m;
    Index kernel_n;
    Index out_channels;
    Index dst_m;
    Index dst_n;
    Scalar beta;
};

// StarPU wrapper for kernel::conv2d_v2_bwd_input_inplace::cpu<T>
template <typename T> void cpu(void *buffers[], void *cl_args) noexcept;

extern Codelet codelet_bf16, codelet_fp32, codelet_fp32_fast_tf32,
       codelet_fp64;

template <typename T> constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template <> constexpr Codelet *codelet<bf16_t>()
{
    return &codelet_bf16;
}

template <> constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template <> constexpr Codelet *codelet<fp32_fast_tf32_t>()
{
    return &codelet_fp32_fast_tf32;
}

template <> constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template <typename T>
void submit(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels, Handle kernel,
        Index dst_m, Index dst_n, Scalar beta, Handle dst);

} // namespace nntile::starpu::conv2d_v2_bwd_input_inplace
