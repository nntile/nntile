/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_attn_output/cuda.cu
 * Accumulate attention outputs on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_attn_output/cuda.hh"
#include "nntile/kernel/cuda.hh"

#include <cuda_runtime.h>

namespace nntile::kernel::accumulate_attn_output
{

namespace
{

using lse_repr_t = typename fp32_t::repr_t;

} // namespace

template<typename T>
static __global__
void cuda_kernel(Index head, Index nelems, const fp32_t *src_lse,
        const T *src_attn, fp32_t *dst_lse, T *dst_attn)
//! Accumulate attention outputs on CUDA
/*! @copydoc nntile::kernel::accumulate_attn_output::cuda
 * */
{
    using Y = typename T::repr_t;
    // Thread layout: one warp per (seq, batch) element (idx),
    // each lane handles a subset of head dimension.
    constexpr int warp_size = 32;
    static_assert(warp_size == warpSize, "Unexpected CUDA warp size");

    const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
    const int warp_in_block = static_cast<int>(threadIdx.x) / warp_size;
    const Index warps_per_block = static_cast<Index>(blockDim.x) / warp_size;
    const Index warp_global =
            static_cast<Index>(blockIdx.x) * warps_per_block
            + static_cast<Index>(warp_in_block);
    const Index idx = warp_global;
    if(idx >= nelems)
    {
        return;
    }

    const unsigned mask = 0xFFFFFFFFu;

    // Load LSEs once per warp and broadcast to all lanes.
    lse_repr_t incoming_lse = lse_repr_t(0);
    if(lane == 0)
    {
        incoming_lse = static_cast<lse_repr_t>(src_lse[idx]);
    }
    incoming_lse = __shfl_sync(mask, incoming_lse, 0);

    // Ignore non-finite result of cuDNN, as it means no influence on output
    const bool src_active = ::isfinite(incoming_lse);
    if(!src_active)
    {
        return;
    }

    lse_repr_t old_lse = lse_repr_t(0);
    if(lane == 0)
    {
        old_lse = static_cast<lse_repr_t>(dst_lse[idx]);
    }
    old_lse = __shfl_sync(mask, old_lse, 0);

    lse_repr_t new_lse = lse_repr_t(0);
    lse_repr_t dst_weight = lse_repr_t(0);
    lse_repr_t src_weight = lse_repr_t(0);
    if(lane == 0)
    {
        const lse_repr_t max_lse =
                old_lse > incoming_lse ? old_lse : incoming_lse;
        const lse_repr_t sum = ::expf(old_lse - max_lse)
                + ::expf(incoming_lse - max_lse);
        new_lse = max_lse + ::logf(sum);
        dst_weight = ::expf(old_lse - new_lse);
        src_weight = ::expf(incoming_lse - new_lse);
        dst_lse[idx] = fp32_t(new_lse);
    }
    new_lse = __shfl_sync(mask, new_lse, 0);
    dst_weight = __shfl_sync(mask, dst_weight, 0);
    src_weight = __shfl_sync(mask, src_weight, 0);

    const Index attn_offset = idx * head;
    for(Index h = static_cast<Index>(lane); h < head; h += warp_size)
    {
        const Y dst_val = static_cast<Y>(dst_attn[attn_offset + h]);
        const Y src_val = static_cast<Y>(src_attn[attn_offset + h]);
        const Y updated =
                static_cast<Y>(dst_weight) * dst_val
                + static_cast<Y>(src_weight) * src_val;
        dst_attn[attn_offset + h] = static_cast<T>(updated);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index head, Index seq, Index batch,
        const fp32_t *src_lse, const T *src_attn, fp32_t *dst_lse, T *dst_attn)
    noexcept
//! Accumulate attention outputs on CUDA
/*! @copydoc nntile::kernel::accumulate_attn_output::cpu
 * */
{
    if(head <= 0 || seq <= 0 || batch <= 0)
    {
        return;
    }

    const Index nelems = seq * batch;
    constexpr int threads = 256;
    const dim3 block_dim(threads);
    constexpr Index warp_size = 32;
    constexpr Index warps_per_block = threads / warp_size;
    const dim3 grid_dim(static_cast<unsigned int>(
            (nelems + warps_per_block - 1) / warps_per_block));
    (cuda_kernel<T>)<<<grid_dim, block_dim, 0, stream>>>(
            head, nelems, src_lse, src_attn, dst_lse, dst_attn);
}

// Explicit instantiation
template
void cuda<fp16_t>(cudaStream_t stream, Index head, Index seq, Index batch,
        const fp32_t *src_lse, const fp16_t *src_attn, fp32_t *dst_lse,
        fp16_t *dst_attn)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index head, Index seq, Index batch,
        const fp32_t *src_lse, const bf16_t *src_attn, fp32_t *dst_lse,
        bf16_t *dst_attn)
    noexcept;

template
void cuda<fp32_t>(cudaStream_t stream, Index head, Index seq, Index batch,
        const fp32_t *src_lse, const fp32_t *src_attn, fp32_t *dst_lse,
        fp32_t *dst_attn)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index head, Index seq, Index batch,
        const fp32_t *src_lse, const fp64_t *src_attn, fp32_t *dst_lse,
        fp64_t *dst_attn)
    noexcept;

} // namespace nntile::kernel::accumulate_attn_output
