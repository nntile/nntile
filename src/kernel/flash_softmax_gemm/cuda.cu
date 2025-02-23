/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_softmax_gemm/cuda.cu
 * CUDA kernel to compute softmax((QK')/sqrt(d))*V using pre-computed maxsumexp
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/flash_softmax_gemm/cuda.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace nntile::kernel::flash_softmax_gemm
{

template<typename T>
__global__ void flash_softmax_gemm_kernel(Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *V, T *A)
{
    using Y = typename T::repr_t;
    // Get global thread indices
    const Index batch_idx = blockIdx.y;
    const Index seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Index tid = threadIdx.x;

    constexpr Y inf = Y{1.0} / Y{0.0};

    // Shared memory for K values and softmax values
    __shared__ Y K_shared[256];  // One row of K at a time
    __shared__ Y softmax_shared[256];  // Softmax values for the block

    if (seq_idx < seq)
    {
        // Calculate base offsets
        // In Fortran order: [head, seq, batch]
        const Index Q_base = head * (seq_idx + seq * batch_idx);
        const Index maxsumexp_base = 2*(seq_idx + seq * batch_idx);
        const Index A_base = head * (seq_idx + seq * batch_idx);

        // Get pre-computed max and sumexp values
        const Y max_val = Y{maxsumexp[maxsumexp_base]};
        const Y sumexp = Y{maxsumexp[maxsumexp_base + 1]};

        // Initialize output accumulator
        Y A_local[256] = {0};  // Assuming head <= 256

        // First compute all softmax values
        for (Index j = 0; j < seq; ++j)
        {
            // Load K values for this j into shared memory
            const Index K_base = head * (j + seq * batch_idx);
            for (Index d = tid; d < head; d += blockDim.x)
            {
                K_shared[d] = Y{K[K_base + d]};
            }
            __syncthreads();

            // Compute dot product Q[h,i,b] * K[h,j,b]
            Y dot_prod = 0;
            for (Index d = 0; d < head; ++d)
            {
                dot_prod += Y{Q[d + Q_base]} * K_shared[d];
            }
            dot_prod *= Y{scale};

            // Apply mask and compute softmax value
            if (mask != nullptr && !mask[j + seq * seq_idx])
            {
                softmax_shared[tid] = 0;
            }
            else
            {
                softmax_shared[tid] = ::exp(dot_prod - max_val) / sumexp;
            }
            __syncthreads();

            // Now multiply by V in chunks to better utilize memory bandwidth
            const Index V_base = head * (j + seq * batch_idx);
            for (Index d_start = 0; d_start < head; d_start += blockDim.x)
            {
                // Load chunk of V into shared memory
                if (d_start + tid < head)
                {
                    K_shared[tid] = Y{V[V_base + d_start + tid]};
                }
                __syncthreads();

                // Multiply and accumulate for this chunk
                const Index chunk_size = ::min(Index(blockDim.x), head - d_start);
                for (Index d = 0; d < chunk_size; ++d)
                {
                    if (d_start + d < head)
                    {
                        A_local[d_start + d] += softmax_shared[tid] * K_shared[d];
                    }
                }
                __syncthreads();
            }
        }

        // Store results
        for (Index d = 0; d < head; ++d)
        {
            A[d + A_base] = T{A_local[d]};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept
{
    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Configure kernel launch parameters
    const int block_size = 256;
    const int blocks_per_seq = (seq + block_size - 1) / block_size;

    dim3 grid(blocks_per_seq, batch);
    dim3 block(block_size);

    // Launch kernel
    flash_softmax_gemm_kernel<T><<<grid, block, 0, stream>>>(batch, seq, head,
        scale, K, Q, mask, maxsumexp, V, A);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_t *K, const fp32_t *Q, const bool_t *mask,
        const fp32_t *maxsumexp, const fp32_t *V, fp32_t *A) noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp64_t *K, const fp64_t *Q, const bool_t *mask,
        const fp64_t *maxsumexp, const fp64_t *V, fp64_t *A) noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const bf16_t *K, const bf16_t *Q, const bool_t *mask,
        const bf16_t *maxsumexp, const bf16_t *V, bf16_t *A) noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_tf32_t *K, const fp32_fast_tf32_t *Q, const bool_t *mask,
        const fp32_fast_tf32_t *maxsumexp, const fp32_fast_tf32_t *V,
        fp32_fast_tf32_t *A) noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_fp16_t *K, const fp32_fast_fp16_t *Q, const bool_t *mask,
        const fp32_fast_fp16_t *maxsumexp, const fp32_fast_fp16_t *V,
        fp32_fast_fp16_t *A) noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_bf16_t *K, const fp32_fast_bf16_t *Q, const bool_t *mask,
        const fp32_fast_bf16_t *maxsumexp, const fp32_fast_bf16_t *V,
        fp32_fast_bf16_t *A) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm
