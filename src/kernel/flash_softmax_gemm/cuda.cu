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

template<typename T, int HEAD_SIZE, int HEAD_BLOCK, int Q_BLOCK, int KV_BLOCK, int KV_SPLIT>
__global__ void flash_softmax_gemm_kernel(Index batch, Index seq,
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

    __shared__ T Q_tile[Q_BLOCK][HEAD_SIZE];    // Store entire Q tile for all head blocks
    __shared__ T KV_tile[KV_BLOCK][HEAD_BLOCK]; // Reuse for both K and V tiles
    __shared__ Y softmax_tile[Q_BLOCK][KV_BLOCK];
    __shared__ bool_t mask_tile[Q_BLOCK][KV_BLOCK];
    __shared__ Y accum[Q_BLOCK][HEAD_SIZE];     // Full accumulator for entire head dimension

    // Initialize accumulator
    for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
        for (int h = threadIdx.y; h < HEAD_SIZE; h += blockDim.y) {
            accum[q][h] = 0;
        }
    }
    __syncthreads();

    // First KV tile - load entire Q and process
    if (kv_block_start < kv_block_end) {
        Index kv_tile_idx = kv_block_start;

        // Load mask tile first
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

        // Initialize softmax_tile to 0
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                softmax_tile[q][kv] = 0;
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Load Q tile for current head block (part of the full Q)
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (q_idx < seq) {
                    const T* Q_base = Q + HEAD_SIZE * (q_idx + seq * batch_idx) + head_offset;
                    for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                        // HEAD_BLOCK is guaranteed to be divisible by 4
                        reinterpret_cast<vec4_t&>(Q_tile[q][head_offset + h]) =
                            *reinterpret_cast<const vec4_t*>(&Q_base[h]);
                    }
                }
            }

            // Load K tile for current head block - only for positions where mask is non-zero
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                if (kv_idx < seq) {
                    // Check if this K row is needed for any query in this block
                    bool needed = false;
                    for (int q = 0; q < Q_BLOCK; ++q) {
                        if (mask_tile[q][kv]) {
                            needed = true;
                            break;
                        }
                    }

                    if (needed) {
                        const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx) + head_offset;
                        for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                            // HEAD_BLOCK is guaranteed to be divisible by 4
                            reinterpret_cast<vec4_t&>(KV_tile[kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&K_base[h]);
                        }
                    }
                }
            }
            __syncthreads();

            // Accumulate partial K'Q for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    const Index kv_idx = kv_tile_idx + kv;
                    const Index q_idx = q_tile_idx * Q_BLOCK + q;

                    if (kv_idx < seq && q_idx < seq && mask_tile[q][kv]) {
                        // Use scalar operations to avoid misalignment
                        Y total = 0;

                        // Process elements one by one to avoid alignment issues
                        for (int h = 0; h < HEAD_BLOCK; h++) {
                            total += Y{KV_tile[kv][h]} * Y{Q_tile[q][head_offset + h]};
                        }

                        softmax_tile[q][kv] += total;
                    }
                }
            }
            __syncthreads();
        }

        // Apply softmax
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const Y max_val = Y{maxsumexp[2 * (q_idx + seq * batch_idx)]};
                const Y sumexp = Y{maxsumexp[2 * (q_idx + seq * batch_idx) + 1]};

                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    if (mask_tile[q][kv]) {
                        softmax_tile[q][kv] = ::exp(Y{scale} * softmax_tile[q][kv] - max_val) / sumexp;
                    } else {
                        softmax_tile[q][kv] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute V @ softmax
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Load V tile for current head block - only for positions where mask is non-zero
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                if (kv_idx < seq) {
                    // Check if this V row is needed for any query in this block
                    bool needed = false;
                    for (int q = 0; q < Q_BLOCK; ++q) {
                        if (mask_tile[q][kv]) {
                            needed = true;
                            break;
                        }
                    }

                    if (needed) {
                        const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx) + head_offset;
                        for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                            // HEAD_BLOCK is guaranteed to be divisible by 4
                            reinterpret_cast<vec4_t&>(KV_tile[kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&V_base[h]);
                        }
                    }
                }
            }
            __syncthreads();

            // Compute V @ softmax for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (q_idx < seq) {
                    for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                        // HEAD_BLOCK is guaranteed to be divisible by 4, so we can always use vector loads
                        vec4_t sum_vec = {0, 0, 0, 0};
                        for (int kv = 0; kv < KV_BLOCK; ++kv) {
                            if (mask_tile[q][kv]) {
                                vec4_t v_vec = reinterpret_cast<const vec4_t&>(KV_tile[kv][h]);
                                Y s = softmax_tile[q][kv];
                                sum_vec.x += v_vec.x * s;
                                sum_vec.y += v_vec.y * s;
                                sum_vec.z += v_vec.z * s;
                                sum_vec.w += v_vec.w * s;
                            }
                        }

                        // Accumulate into the correct position
                        Y* acc_ptr = &accum[q][head_offset + h];
                        acc_ptr[0] += sum_vec.x;
                        acc_ptr[1] += sum_vec.y;
                        acc_ptr[2] += sum_vec.z;
                        acc_ptr[3] += sum_vec.w;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Process remaining K,V tiles - reuse Q from shared memory
    for (Index kv_tile_idx = kv_block_start + KV_BLOCK; kv_tile_idx < kv_block_end; kv_tile_idx += KV_BLOCK) {
        // Load mask tile first
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

        // Initialize softmax_tile to 0
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                softmax_tile[q][kv] = 0;
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Load K tile for current head block - only for positions where mask is non-zero
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                if (kv_idx < seq) {
                    // Check if this K row is needed for any query in this block
                    bool needed = false;
                    for (int q = 0; q < Q_BLOCK; ++q) {
                        if (mask_tile[q][kv]) {
                            needed = true;
                            break;
                        }
                    }

                    if (needed) {
                        const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx) + head_offset;
                        for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                            // HEAD_BLOCK is guaranteed to be divisible by 4
                            reinterpret_cast<vec4_t&>(KV_tile[kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&K_base[h]);
                        }
                    }
                }
            }
            __syncthreads();

            // Accumulate partial K'Q for this head block - optimized version for remaining KV tiles
            // Each thread computes a 2x2 block of the output matrix
            for (int q_base = threadIdx.x * 2; q_base < Q_BLOCK; q_base += blockDim.x * 2) {
                for (int kv_base = threadIdx.y * 2; kv_base < KV_BLOCK; kv_base += blockDim.y * 2) {
                    // Register cache for 2x2 output block
                    Y sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;

                    // Process 4 elements at a time with loop unrolling
                    // HEAD_BLOCK is guaranteed to be divisible by 4
                    #pragma unroll
                    for (int h = 0; h < HEAD_BLOCK; h += 4) {
                        // Load K values for 2 rows
                        Y k00 = Y{KV_tile[kv_base][h]};
                        Y k01 = Y{KV_tile[kv_base][h+1]};
                        Y k02 = Y{KV_tile[kv_base][h+2]};
                        Y k03 = Y{KV_tile[kv_base][h+3]};

                        Y k10 = Y{KV_tile[kv_base+1][h]};
                        Y k11 = Y{KV_tile[kv_base+1][h+1]};
                        Y k12 = Y{KV_tile[kv_base+1][h+2]};
                        Y k13 = Y{KV_tile[kv_base+1][h+3]};

                        // Load Q values for 2 rows
                        Y q00 = Y{Q_tile[q_base][head_offset+h]};
                        Y q01 = Y{Q_tile[q_base][head_offset+h+1]};
                        Y q02 = Y{Q_tile[q_base][head_offset+h+2]};
                        Y q03 = Y{Q_tile[q_base][head_offset+h+3]};

                        Y q10 = Y{Q_tile[q_base+1][head_offset+h]};
                        Y q11 = Y{Q_tile[q_base+1][head_offset+h+1]};
                        Y q12 = Y{Q_tile[q_base+1][head_offset+h+2]};
                        Y q13 = Y{Q_tile[q_base+1][head_offset+h+3]};

                        // Compute partial dot products for 2x2 output block
                        sum00 += k00 * q00 + k01 * q01 + k02 * q02 + k03 * q03;
                        sum01 += k00 * q10 + k01 * q11 + k02 * q12 + k03 * q13;
                        sum10 += k10 * q00 + k11 * q01 + k12 * q02 + k13 * q03;
                        sum11 += k10 * q10 + k11 * q11 + k12 * q12 + k13 * q13;
                    }

                    // Store results to shared memory
                    softmax_tile[q_base][kv_base] += sum00;
                    softmax_tile[q_base+1][kv_base] += sum01;
                    softmax_tile[q_base][kv_base+1] += sum10;
                    softmax_tile[q_base+1][kv_base+1] += sum11;
                }
            }
            __syncthreads();
        }

        // Apply softmax
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const Y max_val = Y{maxsumexp[2 * (q_idx + seq * batch_idx)]};
                const Y sumexp = Y{maxsumexp[2 * (q_idx + seq * batch_idx) + 1]};

                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    if (mask_tile[q][kv]) {
                        softmax_tile[q][kv] = ::exp(Y{scale} * softmax_tile[q][kv] - max_val) / sumexp;
                    } else {
                        softmax_tile[q][kv] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute V @ softmax
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Load V tile for current head block - only for positions where mask is non-zero
            for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                const Index kv_idx = kv_tile_idx + kv;
                if (kv_idx < seq) {
                    // Check if this V row is needed for any query in this block
                    bool needed = false;
                    for (int q = 0; q < Q_BLOCK; ++q) {
                        if (mask_tile[q][kv]) {
                            needed = true;
                            break;
                        }
                    }

                    if (needed) {
                        const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx) + head_offset;
                        for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                            // HEAD_BLOCK is guaranteed to be divisible by 4
                            reinterpret_cast<vec4_t&>(KV_tile[kv][h]) =
                                *reinterpret_cast<const vec4_t*>(&V_base[h]);
                        }
                    }
                }
            }
            __syncthreads();

            // Compute V @ softmax for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (q_idx < seq) {
                    for (int h = threadIdx.y * 4; h < HEAD_BLOCK; h += blockDim.y * 4) {
                        // HEAD_BLOCK is guaranteed to be divisible by 4, so we can always use vector loads
                        vec4_t sum_vec = {0, 0, 0, 0};
                        for (int kv = 0; kv < KV_BLOCK; ++kv) {
                            if (mask_tile[q][kv]) {
                                vec4_t v_vec = reinterpret_cast<const vec4_t&>(KV_tile[kv][h]);
                                Y s = softmax_tile[q][kv];
                                sum_vec.x += v_vec.x * s;
                                sum_vec.y += v_vec.y * s;
                                sum_vec.z += v_vec.z * s;
                                sum_vec.w += v_vec.w * s;
                            }
                        }

                        // Accumulate into the correct position
                        Y* acc_ptr = &accum[q][head_offset + h];
                        acc_ptr[0] += sum_vec.x;
                        acc_ptr[1] += sum_vec.y;
                        acc_ptr[2] += sum_vec.z;
                        acc_ptr[3] += sum_vec.w;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Final atomic accumulation to global memory
    for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
        const Index q_idx = q_tile_idx * Q_BLOCK + q;
        if (q_idx < seq) {
            T* A_base = A + HEAD_SIZE * (q_idx + seq * batch_idx);
            for (int h = threadIdx.y; h < HEAD_SIZE; h += blockDim.y) {
                if constexpr (std::is_same_v<T, fp32_t>) {
                    atomicAdd((float *)&A_base[h], accum[q][h]);
                } else if constexpr (std::is_same_v<T, fp64_t>) {
                    atomicAdd((double *)&A_base[h], accum[q][h]);
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
    constexpr int Q_BLOCK = 32;
    constexpr int KV_BLOCK = 32;
    constexpr int KV_SPLIT = 1;  // Balance between parallelism and overhead

    // For head=64, use 8x8 threads (64 total)
    // For head=128, use 8x16 threads (128 total)
    // For head=256, use 16x16 threads (256 total)
    dim3 threads;
    if (head <= 64) {
        threads = dim3(8, 8);
    } else if (head <= 128) {
        threads = dim3(8, 16);
    } else {
        threads = dim3(16, 16);
    }

    dim3 blocks(KV_SPLIT, (seq + Q_BLOCK - 1) / Q_BLOCK, batch);

    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Clear the output
    cudaMemsetAsync(A, 0, batch * head * seq * sizeof(T), stream);

    // Launch kernel based on head size
    // Note: HEAD_BLOCK must be divisible by 4 for optimal vectorized memory access
    if (head == 64) {
        constexpr int HEAD_SIZE = 64;
        constexpr int HEAD_BLOCK = 16;  // Process in 4 blocks, must be divisible by 4
        flash_softmax_gemm_kernel<T, HEAD_SIZE, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
            <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp, V, A);
    } else if (head == 128) {
        constexpr int HEAD_SIZE = 128;
        constexpr int HEAD_BLOCK = 32;  // Process in 4 blocks, must be divisible by 4
        flash_softmax_gemm_kernel<T, HEAD_SIZE, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
            <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp, V, A);
    } else if (head == 256) {
        constexpr int HEAD_SIZE = 256;
        constexpr int HEAD_BLOCK = 64;  // Process in 4 blocks, must be divisible by 4
        flash_softmax_gemm_kernel<T, HEAD_SIZE, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
            <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp, V, A);
    }
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
