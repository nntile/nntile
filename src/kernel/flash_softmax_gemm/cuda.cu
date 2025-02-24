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

template<typename T, int HEAD_BLOCK, int Q_BLOCK, int KV_BLOCK>
__global__ void flash_softmax_gemm_kernel(Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *V, T *A)
{
    using Y = typename T::repr_t;
    // Block indices
    const Index batch_idx = blockIdx.z;  // Batch index
    const Index q_tile_idx = blockIdx.y; // Q tile index

    // Shared memory for tiles
    __shared__ T Q_tile[HEAD_BLOCK][Q_BLOCK];    // Q tile
    __shared__ T K_tile[HEAD_BLOCK][KV_BLOCK];   // K tile
    __shared__ T V_tile[HEAD_BLOCK][KV_BLOCK];   // V tile
    __shared__ Y softmax_tile[KV_BLOCK][Q_BLOCK]; // K^T @ Q result
    __shared__ bool_t mask_tile[KV_BLOCK][Q_BLOCK]; // Mask tile
    __shared__ Y accum[HEAD_BLOCK][Q_BLOCK];     // Accumulator for V @ softmax

    // Initialize accumulator to zero
    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            accum[h][q] = 0;
        }
    }
    __syncthreads();

    // Load Q tile once - it stays constant for all K,V tiles
    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                Q_tile[h][q] = Q[h + head * (q_idx + seq * batch_idx)];
            }
        }
    }
    __syncthreads();

    // Process all K,V tiles
    for (Index kv_tile_idx = 0; kv_tile_idx < seq; kv_tile_idx += KV_BLOCK) {
        // Load K and V tiles
        for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                if (kv_idx < seq) {
                    K_tile[h][kv] = K[h + head * (kv_idx + seq * batch_idx)];
                    V_tile[h][kv] = V[h + head * (kv_idx + seq * batch_idx)];
                }
            }
        }

        // Load mask tile
        for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (kv_idx < seq && q_idx < seq) {
                    mask_tile[kv][q] = mask[kv_idx + q_idx * seq];
                } else {
                    mask_tile[kv][q] = false;
                }
            }
        }
        __syncthreads();

        // Compute K^T @ Q and apply mask
        for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;

                if (kv_idx < seq && q_idx < seq) {
                    // Compute dot product for this element
                    Y sum = 0;
                    for (int h = 0; h < HEAD_BLOCK; ++h) {
                        sum += Y{K_tile[h][kv]} * Y{Q_tile[h][q]};
                    }

                    // Apply mask and scaling
                    if (mask_tile[kv][q]) {
                        softmax_tile[kv][q] = sum * Y{scale};
                    } else {
                        softmax_tile[kv][q] = -std::numeric_limits<Y>::infinity();
                    }
                }
            }
        }
        __syncthreads();

        // Apply softmax using pre-computed maxsumexp
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const Y max_val = Y{maxsumexp[2 * (q_idx + seq * batch_idx)]};
                const Y sumexp = Y{maxsumexp[2 * (q_idx + seq * batch_idx) + 1]};

                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    if (mask_tile[kv][q]) {
                        softmax_tile[kv][q] = ::exp(softmax_tile[kv][q] - max_val) / sumexp;
                    } else {
                        softmax_tile[kv][q] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // Compute V @ softmax and accumulate
        for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                Y sum = 0;
                for (int kv = 0; kv < KV_BLOCK; ++kv) {
                    if (mask_tile[kv][q]) {
                        sum += Y{V_tile[h][kv]} * softmax_tile[kv][q];
                    }
                }
                accum[h][q] += sum;
            }
        }
        __syncthreads();
    }

    // Write accumulated results to output
    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                A[h + head * (q_idx + seq * batch_idx)] = T{accum[h][q]};
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept
{
    // Define block and grid sizes
    constexpr int HEAD_BLOCK = 64;
    constexpr int Q_BLOCK = 16;
    constexpr int KV_BLOCK = 16;

    dim3 threads(8, 8);  // 64 threads per block
    dim3 blocks(1, (seq + Q_BLOCK - 1) / Q_BLOCK, batch);

    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Clear the output
    //cudaMemsetAsync(A, 0, batch * head * seq * sizeof(T), stream);

    // Launch kernel
    flash_softmax_gemm_kernel<T, HEAD_BLOCK, Q_BLOCK, KV_BLOCK>
        <<<blocks, threads, 0, stream>>>(batch, seq, head, scale, K, Q, mask, maxsumexp, V, A);
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
