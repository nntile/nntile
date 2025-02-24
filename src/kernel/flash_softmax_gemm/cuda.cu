/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_softmax_gemm/cuda.cu
 * CUDA kernel to compute softmax(mask(QK')/sqrt(d))*V using pre-computed maxsumexp
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

template<typename T, int HEAD_BLOCK, int Q_BLOCK, int KV_BLOCK, int KV_SPLIT>
__global__ void flash_softmax_gemm_kernel(Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *V, T *A)
{
    using Y = typename T::repr_t;
    // Use float4 or double4 for vectorized loads
    using vec4_t = typename std::conditional<std::is_same_v<T, fp32_t>, float4, double4>::type;

    // Block indices
    const Index batch_idx = blockIdx.z;
    const Index q_tile_idx = blockIdx.y;
    const Index kv_split_idx = blockIdx.x;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + KV_BLOCK - 1) / KV_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * KV_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    // Shared memory for tiles - double buffer for both K and V
    __shared__ T Q_tile[Q_BLOCK][HEAD_BLOCK];    // Transposed layout
    __shared__ T K_tile[2][KV_BLOCK][HEAD_BLOCK];   // Double buffered, transposed layout
    __shared__ T V_tile[2][KV_BLOCK][HEAD_BLOCK];   // Double buffered, transposed layout
    __shared__ Y softmax_tile[Q_BLOCK][KV_BLOCK]; // Transposed layout [Q][KV]
    __shared__ bool_t mask_tile[Q_BLOCK][KV_BLOCK];  // Transposed layout
    __shared__ Y accum[Q_BLOCK][HEAD_BLOCK];     // Transposed layout

    // Initialize accumulator to zero using vec4
    for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
        for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
            if (h + 3 < HEAD_BLOCK) {
                reinterpret_cast<vec4_t&>(accum[q][h]) = {0, 0, 0, 0};
            } else {
                for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                    accum[q][h + i] = 0;
                }
            }
        }
    }

    // Load Q tile using vec4 along the contiguous HEAD_BLOCK dimension
    for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
        const Index q_idx = q_tile_idx * Q_BLOCK + q;
        if (q_idx < seq) {
            const T* Q_base = Q + head * (q_idx + seq * batch_idx);
            for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                if (h + 3 < HEAD_BLOCK) {
                    reinterpret_cast<vec4_t&>(Q_tile[q][h]) =
                        *reinterpret_cast<const vec4_t*>(&Q_base[h]);
                } else {
                    for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                        Q_tile[q][h + i] = Q_base[h + i];
                    }
                }
            }
        }
    }
    __syncthreads();

    // Process all K,V tiles
    int current_buffer = 0;
    for (Index kv_tile_idx = kv_block_start; kv_tile_idx < kv_block_end;
            kv_tile_idx += KV_BLOCK)
    {
        // Load current K and V tiles
        for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
            const Index kv_idx = kv_tile_idx + kv;
            if (kv_idx < seq) {
                const T* K_base = K + head * (kv_idx + seq * batch_idx);
                const T* V_base = V + head * (kv_idx + seq * batch_idx);

                // Load current K and V tiles
                for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                    if (h + 3 < HEAD_BLOCK) {
                        reinterpret_cast<vec4_t&>(K_tile[current_buffer][kv][h]) =
                            *reinterpret_cast<const vec4_t*>(&K_base[h]);
                        reinterpret_cast<vec4_t&>(V_tile[current_buffer][kv][h]) =
                            *reinterpret_cast<const vec4_t*>(&V_base[h]);
                    } else {
                        for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                            K_tile[current_buffer][kv][h + i] = K_base[h + i];
                            V_tile[current_buffer][kv][h + i] = V_base[h + i];
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Load mask tile
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                const Index kv_idx = kv_tile_idx + kv;
                if (q_idx < seq && kv_idx < seq) {
                    mask_tile[q][kv] = bool{mask[kv_idx + q_idx * seq]};
                } else {
                    mask_tile[q][kv] = false;
                }
            }
        }
        __syncthreads();

        // Start loading next K and V tiles if not last iteration
        if (kv_tile_idx + KV_BLOCK < kv_block_end) {
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv + KV_BLOCK;
                if (kv_idx < seq) {
                    const T* K_next = K + head * (kv_idx + seq * batch_idx);
                    const T* V_next = V + head * (kv_idx + seq * batch_idx);

                    for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                        if (h + 3 < HEAD_BLOCK) {
                            reinterpret_cast<vec4_t&>(K_tile[1-current_buffer][kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&K_next[h]);
                            reinterpret_cast<vec4_t&>(V_tile[1-current_buffer][kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&V_next[h]);
                        } else {
                            for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                                K_tile[1-current_buffer][kv][h + i] = K_next[h + i];
                                V_tile[1-current_buffer][kv][h + i] = V_next[h + i];
                            }
                        }
                    }
                }
            }
        }

        // Compute K^T @ Q using current K buffer
        for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;

                if (kv_idx < seq && q_idx < seq) {
                    vec4_t sum_vec = {0, 0, 0, 0};
                    #pragma unroll
                    for (int h = 0; h < HEAD_BLOCK; h += 4) {
                        if (h + 3 < HEAD_BLOCK) {
                            vec4_t k_vec = reinterpret_cast<const vec4_t&>(K_tile[current_buffer][kv][h]);
                            vec4_t q_vec = reinterpret_cast<const vec4_t&>(Q_tile[q][h]);

                            sum_vec.x += k_vec.x * q_vec.x;
                            sum_vec.y += k_vec.y * q_vec.y;
                            sum_vec.z += k_vec.z * q_vec.z;
                            sum_vec.w += k_vec.w * q_vec.w;
                        } else {
                            for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                                sum_vec.x += Y{K_tile[current_buffer][kv][h + i]} * Y{Q_tile[q][h + i]};
                            }
                        }
                    }

                    Y final_sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

                    if (mask_tile[q][kv]) {
                        softmax_tile[q][kv] = final_sum * Y{scale};
                    } else {
                        softmax_tile[q][kv] = -std::numeric_limits<Y>::infinity();
                    }
                }
            }
        }
        __syncthreads();

        // Apply softmax using transposed layout
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const Y max_val = Y{maxsumexp[2 * (q_idx + seq * batch_idx)]};
                const Y sumexp = Y{maxsumexp[2 * (q_idx + seq * batch_idx) + 1]};

                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    if (mask_tile[q][kv]) {
                        softmax_tile[q][kv] = ::exp(softmax_tile[q][kv] - max_val) / sumexp;
                    } else {
                        softmax_tile[q][kv] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // Compute V @ softmax using current V buffer
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                if (h + 3 < HEAD_BLOCK) {
                    vec4_t sum_vec = {0, 0, 0, 0};
                    for (int kv = 0; kv < KV_BLOCK; ++kv) {
                        if (mask_tile[q][kv]) {
                            vec4_t v_vec = reinterpret_cast<const vec4_t&>(V_tile[current_buffer][kv][h]);
                            Y s = softmax_tile[q][kv];
                            sum_vec.x += v_vec.x * s;
                            sum_vec.y += v_vec.y * s;
                            sum_vec.z += v_vec.z * s;
                            sum_vec.w += v_vec.w * s;
                        }
                    }
                    vec4_t acc_vec = reinterpret_cast<const vec4_t&>(accum[q][h]);
                    acc_vec.x += sum_vec.x;
                    acc_vec.y += sum_vec.y;
                    acc_vec.z += sum_vec.z;
                    acc_vec.w += sum_vec.w;
                    reinterpret_cast<vec4_t&>(accum[q][h]) = acc_vec;
                } else {
                    // Handle boundary case
                    for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                        Y sum = 0;
                        for (int kv = 0; kv < KV_BLOCK; ++kv) {
                            if (mask_tile[q][kv]) {
                                sum += Y{V_tile[current_buffer][kv][h + i]} * softmax_tile[q][kv];
                            }
                        }
                        accum[q][h + i] += sum;
                    }
                }
            }
        }
        __syncthreads();

        // Swap buffers for next iteration
        current_buffer = 1 - current_buffer;
    }

    // Write accumulated results using vectorized loads and scalar atomic adds
    for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
        const Index q_idx = q_tile_idx * Q_BLOCK + q;
        if (q_idx < seq) {
            T* A_base = A + head * (q_idx + seq * batch_idx);
            for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                if (h + 3 < HEAD_BLOCK) {
                    // Load 4 values at once from accumulator
                    vec4_t acc_vec = reinterpret_cast<const vec4_t&>(accum[q][h]);

                    // Perform atomic adds for each component
                    if constexpr (std::is_same_v<T, fp32_t>) {
                        atomicAdd((float *)&A_base[h], acc_vec.x);
                        atomicAdd((float *)&A_base[h + 1], acc_vec.y);
                        atomicAdd((float *)&A_base[h + 2], acc_vec.z);
                        atomicAdd((float *)&A_base[h + 3], acc_vec.w);
                    } else if constexpr (std::is_same_v<T, fp64_t>) {
                        atomicAdd((double *)&A_base[h], acc_vec.x);
                        atomicAdd((double *)&A_base[h + 1], acc_vec.y);
                        atomicAdd((double *)&A_base[h + 2], acc_vec.z);
                        atomicAdd((double *)&A_base[h + 3], acc_vec.w);
                    }
                } else {
                    // Handle boundary case
                    for (int i = 0; i < 4 && h + i < HEAD_BLOCK; ++i) {
                        if constexpr (std::is_same_v<T, fp32_t>) {
                            atomicAdd((float *)&A_base[h + i], accum[q][h + i]);
                        } else if constexpr (std::is_same_v<T, fp64_t>) {
                            atomicAdd((double *)&A_base[h + i], accum[q][h + i]);
                        }
                    }
                }
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
    constexpr int KV_BLOCK = 32;
    constexpr int KV_SPLIT = 16;

    dim3 threads(8, 8);  // 256 threads per block
    dim3 blocks(KV_SPLIT, (seq + Q_BLOCK - 1) / Q_BLOCK, batch);

    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Clear the output
    cudaMemsetAsync(A, 0, batch * head * seq * sizeof(T), stream);

    // Launch kernel
    flash_softmax_gemm_kernel<T, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
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
