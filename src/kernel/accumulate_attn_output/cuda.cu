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
    const Index idx = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= nelems)
    {
        return;
    }

    const lse_repr_t incoming_lse = static_cast<lse_repr_t>(src_lse[idx]);
    // Ignore non-finite result of cuDNN, as it means no influence on output
    const bool src_active = ::isfinite(incoming_lse);

    if(!src_active)
    {
        return;
    }

    const lse_repr_t old_lse = static_cast<lse_repr_t>(dst_lse[idx]);
    lse_repr_t max_lse, sum;
    Y dst_weight, src_weight;
    if(old_lse > incoming_lse)
    {
        max_lse = old_lse;
        const lse_repr_t exp_diff = ::expf(incoming_lse - max_lse);
        sum = lse_repr_t(1.0) + exp_diff;
        dst_weight = Y(lse_repr_t(1.0) / sum);
        src_weight = Y(exp_diff) * dst_weight;
    }
    else
    {
        max_lse = incoming_lse;
        const lse_repr_t exp_diff = ::expf(old_lse - max_lse);
        sum = lse_repr_t(1.0) + exp_diff;
        src_weight = Y(lse_repr_t(1.0) / sum);
        dst_weight = Y(exp_diff) * src_weight;
    }
    dst_lse[idx] = fp32_t(max_lse + ::logf(sum));

    const Index attn_offset = idx * head;
    constexpr size_t vector_size = sizeof(float4) / sizeof(T);

    // If head is a multiple of vector_size, use vectorized approach
    if (head % vector_size == 0)
    {
        // Prefetch dst and src vectors
        float4 *dst_vector_ptr = reinterpret_cast<float4 *>(dst_attn + attn_offset);
        float4 dst_vector = dst_vector_ptr[0];
        float4 src_vector = reinterpret_cast<const float4 *>(src_attn + attn_offset)[0];
        for(Index h = 0; h < head; h += vector_size)
        {
            #pragma unroll vector_size
            for(Index i = 0; i < vector_size; ++i)
            {
                const Y dst_val = static_cast<Y>(reinterpret_cast<T *>(&dst_vector)[i]);
                const Y src_val = static_cast<Y>(reinterpret_cast<T *>(&src_vector)[i]);
                const Y updated = dst_weight * dst_val + src_weight * src_val;
                reinterpret_cast<T *>(&dst_vector)[i] = static_cast<T>(updated);
            }
            dst_vector_ptr[0] = dst_vector;
            // Prefetch next dst and src vectors
            if(h + vector_size < head)
            {
                dst_vector_ptr = reinterpret_cast<float4 *>(dst_attn + attn_offset + h + vector_size);
                dst_vector = dst_vector_ptr[0];
                src_vector = reinterpret_cast<const float4 *>(src_attn + attn_offset + h + vector_size)[0];
            }
        }
    }
    else
    {
        for(Index h = 0; h < head; ++h)
        {
            const Y dst_val = static_cast<Y>(dst_attn[attn_offset + h]);
            const Y src_val = static_cast<Y>(src_attn[attn_offset + h]);
            const Y updated = dst_weight * dst_val + src_weight * src_val;
            dst_attn[attn_offset + h] = static_cast<T>(updated);
        }
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
    constexpr int threads = 32;
    const dim3 block_dim(threads);
    const dim3 grid_dim(static_cast<unsigned int>(
            (nelems + threads - 1) / threads));
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
