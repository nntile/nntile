/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_softmax_gemm_backward_dq_dk/cuda.cu
 * CUDA kernel to compute gradients dQ and dK of softmax(A)V
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/flash_softmax_gemm_backward_dq_dk/cuda.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
{

template<typename T>
__global__ void flash_softmax_gemm_backward_dq_dk_kernel(
        Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *dA, const T *V,
        const T *sumprod_slice, T *dQ, T *dK)
{
    using Y = typename T::repr_t;
    // Get global thread indices
    const Index batch_idx = blockIdx.y;
    const Index seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Index tid = threadIdx.x;

    constexpr Y inf = Y{1.0} / Y{0.0};

    // Shared memory for K, V and softmax values
    __shared__ Y K_shared[256];  // One row of K at a time
    __shared__ Y V_shared[256];  // One row of V at a time
    __shared__ Y softmax_shared[256];  // Softmax values for the block

    if (seq_idx < seq)
    {
        // Calculate base offsets
        // In Fortran order: [head, seq, batch]
        const Index Q_base = head * (seq_idx + seq * batch_idx);
        const Index maxsumexp_base = 2*(seq_idx + seq * batch_idx);

        // Get pre-computed max and sumexp values
        const Y max_val = Y{maxsumexp[maxsumexp_base]};
        const Y sumexp = Y{maxsumexp[maxsumexp_base + 1]};
        const Y sumprod = Y{sumprod_slice[seq_idx + seq * batch_idx]};

        // First compute all softmax values and gradients
        for (Index j = 0; j < seq; ++j)
        {
            // Load K values for this j into shared memory
            const Index K_base = head * (j + seq * batch_idx);
            const Index V_base = head * (j + seq * batch_idx);
            for (Index d = tid; d < head; d += blockDim.x)
            {
                K_shared[d] = Y{K[K_base + d]};
                V_shared[d] = Y{V[V_base + d]};
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
            Y softmax_val;
            if (mask != nullptr && !mask[j + seq * seq_idx])
            {
                softmax_val = 0;
            }
            else
            {
                softmax_val = ::exp(dot_prod - max_val) / sumexp;
            }
            softmax_shared[tid] = softmax_val;
            __syncthreads();

            // First compute V^T * dA into shared memory (like tmp_grad)
            const Index dA_base = head * (seq_idx + seq * batch_idx);  // [head, seq_i, batch]
            Y tmp_grad = 0;
            for (Index d = 0; d < head; ++d) {
                tmp_grad += V_shared[d] * Y{dA[d + dA_base]};
            }
            __syncthreads();

            // Subtract sumprod and multiply by softmax
            // sumprod_slice is [seq, batch] in Fortran order
            const Index sumprod_idx = seq_idx + seq * batch_idx;
            tmp_grad = (tmp_grad - Y{sumprod_slice[sumprod_idx]}) * softmax_shared[tid];

            // Apply mask
            if (mask != nullptr && !mask[j + seq * seq_idx]) {
                tmp_grad = 0;
            }

            // Now compute gradients for Q and K using tmp_grad
            for (Index d_start = 0; d_start < head; d_start += blockDim.x) {
                // Load chunk of K and Q into shared memory
                if (d_start + tid < head) {
                    // For dQ we need K^T, so load K[h,j,b]
                    K_shared[tid] = Y{K[d_start + tid + head * (j + seq * batch_idx)]};
                    // For dK we need Q, so load Q[h,i,b]
                    V_shared[tid] = Y{Q[d_start + tid + head * (seq_idx + seq * batch_idx)]};
                }
                __syncthreads();

                const Index chunk_size = ::min(Index(blockDim.x), head - d_start);
                for (Index d = 0; d < chunk_size; ++d) {
                    if (d_start + d < head) {
                        // Update dQ[h,i,b] = tmp_grad[i,j] * K[h,j,b] / sqrt(d)
                        // For each j, accumulate into dQ[h,i,b]
                        const Index dQ_idx = (d_start + d) + head * (seq_idx + seq * batch_idx);
                        Y grad_q = Y{scale} * tmp_grad * K_shared[d];
                        if constexpr (std::is_same_v<Y, float>) {
                            atomicAdd(reinterpret_cast<float*>(&dQ[dQ_idx]), grad_q);
                        } else if constexpr (std::is_same_v<Y, double>) {
                            atomicAdd(reinterpret_cast<double*>(&dQ[dQ_idx]), grad_q);
                        }

                        // Update dK[h,j,b] = tmp_grad[i,j] * Q[h,i,b] / sqrt(d)
                        // For each i, accumulate into dK[h,j,b]
                        const Index dK_idx = (d_start + d) + head * (j + seq * batch_idx);
                        Y grad_k = Y{scale} * tmp_grad * V_shared[d];  // V_shared contains Q
                        if constexpr (std::is_same_v<Y, float>) {
                            atomicAdd(reinterpret_cast<float*>(&dK[dK_idx]), grad_k);
                        } else if constexpr (std::is_same_v<Y, double>) {
                            atomicAdd(reinterpret_cast<double*>(&dK[dK_idx]), grad_k);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *dA, const T *V, const T *sumprod_slice,
          T *dQ, T *dK) noexcept
{
    // Calculate scaling factor
    using Y = typename T::repr_t;
    Y scale_val = Y(1.0) / std::sqrt(Y(head));
    T scale = T{scale_val};

    // Configure kernel launch parameters
    const int block_size = 256;
    const int blocks_per_seq = (seq + block_size - 1) / block_size;

    dim3 grid(blocks_per_seq, batch);
    dim3 block(block_size);

    // Launch kernel
    flash_softmax_gemm_backward_dq_dk_kernel<T><<<grid, block, 0, stream>>>(
        batch, seq, head, scale, K, Q, mask, maxsumexp,
        dA, V, sumprod_slice, dQ, dK);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_t *K, const fp32_t *Q, const bool_t *mask,
        const fp32_t *maxsumexp, const fp32_t *dA, const fp32_t *V,
        const fp32_t *sumprod_slice, fp32_t *dQ, fp32_t *dK) noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp64_t *K, const fp64_t *Q, const bool_t *mask,
        const fp64_t *maxsumexp, const fp64_t *dA, const fp64_t *V,
        const fp64_t *sumprod_slice, fp64_t *dQ, fp64_t *dK) noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const bf16_t *K, const bf16_t *Q, const bool_t *mask,
        const bf16_t *maxsumexp, const bf16_t *dA, const bf16_t *V,
        const bf16_t *sumprod_slice, bf16_t *dQ, bf16_t *dK) noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_tf32_t *K, const fp32_fast_tf32_t *Q,
        const bool_t *mask, const fp32_fast_tf32_t *maxsumexp,
        const fp32_fast_tf32_t *dA, const fp32_fast_tf32_t *V,
        const fp32_fast_tf32_t *sumprod_slice,
        fp32_fast_tf32_t *dQ, fp32_fast_tf32_t *dK) noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_fp16_t *K, const fp32_fast_fp16_t *Q,
        const bool_t *mask, const fp32_fast_fp16_t *maxsumexp,
        const fp32_fast_fp16_t *dA, const fp32_fast_fp16_t *V,
        const fp32_fast_fp16_t *sumprod_slice,
        fp32_fast_fp16_t *dQ, fp32_fast_fp16_t *dK) noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_bf16_t *K, const fp32_fast_bf16_t *Q,
        const bool_t *mask, const fp32_fast_bf16_t *maxsumexp,
        const fp32_fast_bf16_t *dA, const fp32_fast_bf16_t *V,
        const fp32_fast_bf16_t *sumprod_slice,
        fp32_fast_bf16_t *dQ, fp32_fast_bf16_t *dK) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
