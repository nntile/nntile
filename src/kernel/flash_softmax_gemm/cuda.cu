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


/**
 * Optimized shared memory matrix multiplication: C += A * B^T
 *
 * This function assumes all matrices are already in shared memory.
 * B is transposed during multiplication (A * B^T).
 * All matrices are in row-major format.
 * No bounds checking is performed as sizes are guaranteed to be multiples of 4 or 8.
 * Uses double buffering to overlap computation with memory loads.
 *
 * @tparam T Data type for input matrices A and B
 * @tparam Y Data type for output matrix C
 * @tparam M Number of rows in A and C
 * @tparam N Number of columns in B^T and C
 * @tparam K Number of columns in A and rows in B^T
 * @tparam ldA Leading dimension of A (stride between rows)
 * @tparam ldB Leading dimension of B (stride between rows)
 * @tparam ldC Leading dimension of C (stride between rows)
 * @param A Input matrix A in shared memory (M x ldA)
 * @param B Input matrix B in shared memory (N x ldB)
 * @param C Output matrix C in shared memory (M x ldC)
 */
template<typename T, typename Y, int M, int N, int K, int ldA, int ldB, int ldC>
__device__ void gemm_smem_NT(const Y *A, const T *B, Y *C) {
    // Thread block dimensions
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Number of threads in each dimension
    const int THREAD_X = blockDim.x;
    const int THREAD_Y = blockDim.y;

    // Each thread computes a 2x2 block of the output
    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 2;

    // Loop over output tiles
    for (int m_base = ty * THREAD_TILE_M; m_base < M; m_base += THREAD_Y * THREAD_TILE_M) {
        for (int n_base = tx * THREAD_TILE_N; n_base < N; n_base += THREAD_X * THREAD_TILE_N) {
            // Registers for accumulation
            Y sum[THREAD_TILE_M][THREAD_TILE_N] = {0};

            // Double buffering registers for A and B
            Y a_vals[2][THREAD_TILE_M];
            Y b_vals[2][THREAD_TILE_N];

            // Preload first batch of data
            int buf_idx = 0;
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                const int m_idx = m_base + m_offset;
                a_vals[buf_idx][m_offset] = Y{A[m_idx * ldA + 0]};
            }

            #pragma unroll
            for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                const int n_idx = n_base + n_offset;
                b_vals[buf_idx][n_offset] = Y{B[n_idx * ldB + 0]};
            }

            // Loop over inner dimension with aggressive unrolling and double buffering
            #pragma unroll 4
            for (int k = 0; k < K-1; ++k) {
                // Swap buffers
                buf_idx = 1 - buf_idx;

                // Load next batch of data while computing with current batch
                #pragma unroll
                for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                    const int m_idx = m_base + m_offset;
                    a_vals[buf_idx][m_offset] = Y{A[m_idx * ldA + (k+1)]};
                }

                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    const int n_idx = n_base + n_offset;
                    b_vals[buf_idx][n_offset] = Y{B[n_idx * ldB + (k+1)]};
                }

                // Compute outer product with current data
                #pragma unroll
                for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                    #pragma unroll
                    for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                        sum[m_offset][n_offset] += a_vals[1-buf_idx][m_offset] * b_vals[1-buf_idx][n_offset];
                    }
                }
            }

            // Process the last iteration (no need to load more data)
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    sum[m_offset][n_offset] += a_vals[buf_idx][m_offset] * b_vals[buf_idx][n_offset];
                }
            }

            // Store results to C
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                const int m_idx = m_base + m_offset;
                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    const int n_idx = n_base + n_offset;
                    C[m_idx * ldC + n_idx] += sum[m_offset][n_offset];
                }
            }
        }
    }
}

/**
 * Optimized shared memory matrix multiplication: C += A^T * B
 *
 * This function assumes all matrices are already in shared memory.
 * A is transposed during multiplication (A^T * B).
 * All matrices are in row-major format.
 * Uses double buffering to overlap computation with memory loads.
 *
 * @tparam T Data type for input matrices A and B
 * @tparam Y Data type for output matrix C
 * @tparam M Number of rows in A^T and C
 * @tparam N Number of columns in B and C
 * @tparam K Number of columns in A^T and rows in B
 * @tparam ldA Leading dimension of A (stride between rows)
 * @tparam ldB Leading dimension of B (stride between rows)
 * @tparam ldC Leading dimension of C (stride between rows)
 * @param A Input matrix A in shared memory (K x ldA)
 * @param B Input matrix B in shared memory (K x ldB)
 * @param C Output matrix C in shared memory (M x ldC)
 */
template<typename T, typename Y, int M, int N, int K, int ldA, int ldB, int ldC>
__device__ void gemm_smem_TN(const T *A, const T *B, Y *C) {
    // Thread block dimensions
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Number of threads in each dimension
    const int THREAD_X = blockDim.x;
    const int THREAD_Y = blockDim.y;

    // Each thread computes a 4x2 block of the output
    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 2;

    // Loop over output tiles
    for (int m_base = ty * THREAD_TILE_M; m_base < M; m_base += THREAD_Y * THREAD_TILE_M) {
        for (int n_base = tx * THREAD_TILE_N; n_base < N; n_base += THREAD_X * THREAD_TILE_N) {
            // Registers for accumulation
            Y sum[THREAD_TILE_M][THREAD_TILE_N] = {0};

            // Double buffering registers for A and B
            Y a_vals[2][THREAD_TILE_M];
            Y b_vals[2][THREAD_TILE_N];

            // Preload first batch of data
            int buf_idx = 0;
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                const int m_idx = m_base + m_offset;
                // Access A with transpose: A[0][m_idx] becomes A[0 * ldA + m_idx]
                a_vals[buf_idx][m_offset] = Y{A[0 * ldA + m_idx]};
            }

            #pragma unroll
            for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                const int n_idx = n_base + n_offset;
                b_vals[buf_idx][n_offset] = Y{B[0 * ldB + n_idx]};
            }

            // Loop over inner dimension with aggressive unrolling and double buffering
            #pragma unroll 4
            for (int k = 0; k < K-1; ++k) {
                // Swap buffers
                buf_idx = 1 - buf_idx;

                // Load next batch of data while computing with current batch
                #pragma unroll
                for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                    const int m_idx = m_base + m_offset;
                    // Access A with transpose: A[k+1][m_idx] becomes A[(k+1) * ldA + m_idx]
                    a_vals[buf_idx][m_offset] = Y{A[(k+1) * ldA + m_idx]};
                }

                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    const int n_idx = n_base + n_offset;
                    b_vals[buf_idx][n_offset] = Y{B[(k+1) * ldB + n_idx]};
                }

                // Compute outer product with current data
                #pragma unroll
                for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                    #pragma unroll
                    for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                        sum[m_offset][n_offset] += a_vals[1-buf_idx][m_offset] * b_vals[1-buf_idx][n_offset];
                    }
                }
            }

            // Process the last iteration (no need to load more data)
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    sum[m_offset][n_offset] += a_vals[buf_idx][m_offset] * b_vals[buf_idx][n_offset];
                }
            }

            // Store results to C
            #pragma unroll
            for (int m_offset = 0; m_offset < THREAD_TILE_M; ++m_offset) {
                const int m_idx = m_base + m_offset;
                #pragma unroll
                for (int n_offset = 0; n_offset < THREAD_TILE_N; ++n_offset) {
                    const int n_idx = n_base + n_offset;
                    C[m_idx * ldC + n_idx] += sum[m_offset][n_offset];
                }
            }
        }
    }
}

template<typename T, int HEAD_SIZE, int HEAD_BLOCK, int Q_BLOCK, int KV_BLOCK, int KV_SPLIT>
__global__ void flash_softmax_gemm_kernel(Index batch, Index seq,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *V, T *A)
{
    using Y = typename T::repr_t;

    // Block indices
    const Index batch_idx = blockIdx.y;
    const Index q_tile_idx = blockIdx.x;
    const Index kv_split_idx = blockIdx.z;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + KV_BLOCK - 1) / KV_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * KV_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    __shared__ T Q_tile[2][HEAD_BLOCK][Q_BLOCK+1];    // Transposed Q tile for current head block only
    __shared__ T KV_tile[2][HEAD_BLOCK][KV_BLOCK+1]; // Double buffered KV tile for better memory access
    __shared__ Y softmax_tile[Q_BLOCK][KV_BLOCK+1]; // Will also encode mask information
    __shared__ Y accum[Q_BLOCK][HEAD_BLOCK+1];    // Reduced size accumulator for current head block

    // Process K,V tiles
    for (Index kv_tile_idx = kv_block_start; kv_tile_idx < kv_block_end; kv_tile_idx += KV_BLOCK) {
        // Initialize buffer index for double buffering
        int buf_idx = 0;

        // When loading mask, set softmax_tile to 0 or -infinity based on mask
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                const Index kv_idx = kv_tile_idx + kv;
                if (q_idx < seq && kv_idx < seq && bool{mask[kv_idx + q_idx * seq]}) {
                    softmax_tile[q][kv] = 0; // Valid position, initialize to 0
                } else {
                    softmax_tile[q][kv] = -std::numeric_limits<Y>::infinity(); // Invalid position
                }
            }
        }
        __syncthreads();

        // Load starting K tile for starting head block into the current buffer
        for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
            const Index kv_idx = kv_tile_idx + kv;
            if (kv_idx < seq) {
                // When checking if K row is needed, use is_finite instead of mask_tile
                bool needed = false;
                for (int q = 0; q < Q_BLOCK; ++q) {
                    if (std::isfinite(softmax_tile[q][kv])) {
                        needed = true;
                        break;
                    }
                }

                if (needed) {
                    const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx);
                    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                        KV_tile[buf_idx][h][kv] = K_base[h];
                    }
                }
            }
        }

        // Load starting Q tile for starting head block into the current buffer
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            const T* Q_base = Q + HEAD_SIZE * (q_idx + seq * batch_idx);
            for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                Q_tile[buf_idx][h][q] = Q_base[h];
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK)
        {
            // Prefetch next Q and K tiles if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE) {
                // K tile prefetch
                int next_buf_idx = 1 - buf_idx;
                for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                    const Index kv_idx = kv_tile_idx + kv;
                    if (kv_idx < seq) {
                        bool needed = false;
                        for (int q = 0; q < Q_BLOCK; ++q) {
                            if (std::isfinite(softmax_tile[q][kv])) {
                                needed = true;
                                break;
                            }
                        }

                        if (needed) {
                            const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                            for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                                KV_tile[next_buf_idx][h][kv] = K_base[h];
                            }
                        }
                    }
                }
                // Q tile prefetch
                for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                    const Index q_idx = q_tile_idx * Q_BLOCK + q;
                    if (q_idx < seq) {
                        const T* Q_base = Q + HEAD_SIZE * (q_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                        for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                            Q_tile[next_buf_idx][h][q] = Q_base[h];
                        }
                    }
                }
            }
            // At the last iteration prefetch the first (head_offset==0) V tile
            else {
                int next_buf_idx = 1 - buf_idx;
                for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                    const Index kv_idx = kv_tile_idx + kv;
                    if (kv_idx < seq) {
                        bool needed = false;
                        for (int q = 0; q < Q_BLOCK; ++q) {
                            if (std::isfinite(softmax_tile[q][kv])) {
                                needed = true;
                                break;
                            }
                        }

                        if (needed) {
                            const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx);
                            for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                                KV_tile[next_buf_idx][h][kv] = V_base[h];
                            }
                        }
                    }
                }
            }

            // Accumulate partial K'Q for this head block - optimized version for remaining KV tiles
            // Using gemm_smem_TN to compute softmax_tile = Q_tile^T * KV_tile
            // For gemm_smem_TN(A, B, C) computing C = A^T * B:
            // - A is Q_tile[head_offset:head_offset+HEAD_BLOCK][0:Q_BLOCK]
            // - B is KV_tile[0:HEAD_BLOCK][0:KV_BLOCK]
            // - C is softmax_tile[0:Q_BLOCK][0:KV_BLOCK]
            // - M (rows of A^T and C) = Q_BLOCK
            // - N (cols of B and C) = KV_BLOCK
            // - K (cols of A^T and rows of B) = HEAD_BLOCK
            // - ldA = Q_BLOCK+1 (stride between rows of Q_tile)
            // - ldB = KV_BLOCK+1 (stride between rows of KV_tile)
            // - ldC = KV_BLOCK+1 (stride between rows of softmax_tile)
            gemm_smem_TN<T, Y, Q_BLOCK, KV_BLOCK, HEAD_BLOCK, Q_BLOCK+1, KV_BLOCK+1, KV_BLOCK+1>(
                &Q_tile[buf_idx][0][0],           // A matrix - now starting at 0 since we only have current head block
                &KV_tile[buf_idx][0][0], // B matrix - use current buffer
                &softmax_tile[0][0]      // C matrix
            );
            __syncthreads();

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
        }

        // Apply softmax
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const Y max_val = Y{maxsumexp[2 * (q_idx + seq * batch_idx)]};
                const Y sumexp = Y{maxsumexp[2 * (q_idx + seq * batch_idx) + 1]};

                for (int kv = threadIdx.y; kv < KV_BLOCK; kv += blockDim.y) {
                    if (std::isfinite(softmax_tile[q][kv])) {
                        softmax_tile[q][kv] = ::exp(Y{scale} * softmax_tile[q][kv] - max_val) / sumexp;
                    } else {
                        softmax_tile[q][kv] = 0; // -infinity becomes 0 after softmax
                    }
                }
            }
        }
        __syncthreads();

        // Process head dimension in blocks to compute V @ softmax
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Initialize the smaller accumulator for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                    accum[q][h] = 0;
                }
            }
            __syncthreads();

            // Prefetch next V tile if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE) {
                int next_buf_idx = 1 - buf_idx;
                for (int kv = threadIdx.x; kv < KV_BLOCK; kv += blockDim.x) {
                    const Index kv_idx = kv_tile_idx + kv;
                    if (kv_idx < seq) {
                        bool needed = false;
                        for (int q = 0; q < Q_BLOCK; ++q) {
                            if (std::isfinite(softmax_tile[q][kv])) {
                                needed = true;
                                break;
                            }
                        }

                        if (needed) {
                            const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                            for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                                KV_tile[next_buf_idx][h][kv] = V_base[h];
                            }
                        }
                    }
                }
            }

            // Compute V @ softmax for this head block using optimized gemm_smem_NT
            // For gemm_smem_NT(A, B, C) computing C += A * B^T:
            // - A is softmax_tile[0:Q_BLOCK][0:KV_BLOCK]
            // - B is KV_tile[0:HEAD_BLOCK][0:KV_BLOCK]
            // - C is accum[0:Q_BLOCK][0:HEAD_BLOCK]
            // - M (rows of A and C) = Q_BLOCK
            // - N (cols of B^T and C) = HEAD_BLOCK
            // - K (cols of A and rows of B^T) = KV_BLOCK
            // - ldA = KV_BLOCK+1 (stride between rows of softmax_tile)
            // - ldB = KV_BLOCK+1 (stride between rows of KV_tile)
            // - ldC = HEAD_BLOCK+1 (stride between rows of accum)
            gemm_smem_NT<T, Y, Q_BLOCK, HEAD_BLOCK, KV_BLOCK, KV_BLOCK+1, KV_BLOCK+1, HEAD_BLOCK+1>(
                &softmax_tile[0][0],  // A matrix - softmax values
                &KV_tile[buf_idx][0][0],       // B matrix - V values (will be transposed)
                &accum[0][0]          // C matrix - accumulator for output
            );
            __syncthreads();

            // Atomic accumulation to global memory for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (q_idx < seq) {
                    T* A_base = A + HEAD_SIZE * (q_idx + seq * batch_idx) + head_offset;
                    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                        if constexpr (std::is_same_v<T, fp32_t>) {
                            atomicAdd((float *)&A_base[h], accum[q][h]);
                        } else if constexpr (std::is_same_v<T, fp64_t>) {
                            atomicAdd((double *)&A_base[h], accum[q][h]);
                        }
                    }
                }
            }
            __syncthreads(); // Ensure all threads are done with accum before reusing it

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept
{
    // Define block and grid sizes
    constexpr int Q_BLOCK = 64;
    constexpr int KV_BLOCK = 64;
    constexpr int KV_SPLIT = 4;  // Balance between parallelism and overhead

    // For head=64, use 8x8 threads (64 total)
    // For head=128, use 8x16 threads (128 total)
    // For head=256, use 16x16 threads (256 total)
    dim3 threads;
    if (head <= 64) {
        threads = dim3(16, 16);
    } else if (head <= 128) {
        threads = dim3(8, 16);
    } else {
        threads = dim3(16, 16);
    }

    dim3 blocks((seq + Q_BLOCK - 1) / Q_BLOCK, batch, KV_SPLIT);

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
    } // TODO: enable other heads later
    // } else if (head == 128) {
    //     constexpr int HEAD_SIZE = 128;
    //     constexpr int HEAD_BLOCK = 8;  // Process in 4 blocks, must be divisible by 4
    //     flash_softmax_gemm_kernel<T, HEAD_SIZE, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
    //         <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp, V, A);
    // } else if (head == 256) {
    //     constexpr int HEAD_SIZE = 256;
    //     constexpr int HEAD_BLOCK = 8;  // Process in 4 blocks, must be divisible by 4
    //     flash_softmax_gemm_kernel<T, HEAD_SIZE, HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT>
    //         <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp, V, A);
    // }
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
