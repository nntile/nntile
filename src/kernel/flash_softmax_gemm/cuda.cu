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

template<typename T, Index HEAD_SIZE, Index HEAD_BLOCK, Index Q_BLOCK, Index KV_BLOCK, Index KV_SPLIT>
__global__ void flash_softmax_gemm_kernel(
    Index batch, Index seq, T scale,
    const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
    const T *V, T *A)
{
    using namespace std;
    using Y = typename T::repr_t;

    // Get global indices
    const Index batch_idx = blockIdx.y;
    const Index q_tile_idx = blockIdx.x;
    const Index kv_split_idx = blockIdx.z;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + KV_BLOCK - 1) / KV_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * KV_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    // Define thread-local dimensions for softmax tile
    constexpr int THREAD_Q_BLOCK = 4; // Each thread handles 4 rows
    constexpr int THREAD_KV_BLOCK = 4; // Each thread handles 4 columns

    // Thread indices for accessing the softmax tile
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int q_start = tx * THREAD_Q_BLOCK;
    const int kv_start = ty * THREAD_KV_BLOCK;

    // Shared memory allocations
    __shared__ T Q_tile[2][HEAD_BLOCK][Q_BLOCK+1];    // Double buffered Q tile
    __shared__ T KV_tile[2][HEAD_BLOCK][KV_BLOCK+1];  // Double buffered KV tile
    __shared__ Y accum[Q_BLOCK][HEAD_BLOCK+1];        // Accumulator for current head block
    __shared__ bool is_needed[KV_BLOCK];

    // Thread-local registers for softmax tile
    Y softmax_reg[THREAD_Q_BLOCK][THREAD_KV_BLOCK];
    bool is_needed_reg[THREAD_KV_BLOCK];

    // Process K,V tiles
    for (Index kv_tile_idx = kv_block_start; kv_tile_idx < kv_block_end; kv_tile_idx += KV_BLOCK) {
        // Initialize buffer index for double buffering
        int buf_idx = 0;

        // Initialize softmax registers with mask information
        for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q_start + q_offset;
            for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
                const Index kv_idx = kv_tile_idx + kv_start + kv_offset;
                if (q_idx < seq && kv_idx < seq && bool{mask[kv_idx + q_idx * seq]}) {
                    softmax_reg[q_offset][kv_offset] = 0; // Valid position
                } else {
                    softmax_reg[q_offset][kv_offset] = -std::numeric_limits<Y>::infinity(); // Invalid
                }
            }
        }
        __syncthreads();

        // Clear is_needed flags
        for (int kv = threadIdx.x + threadIdx.y * blockDim.x;
                kv < KV_BLOCK;
                kv += blockDim.x * blockDim.y) {
            is_needed[kv] = false;
        }
        __syncthreads();

        // Mark which K rows are needed
        for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
            const int kv = kv_start + kv_offset;
            if (kv < KV_BLOCK) {
                for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
                    if (std::isfinite(softmax_reg[q_offset][kv_offset])) {
                        is_needed[kv] = true;
                        break;
                    }
                }
            }
        }
        __syncthreads();

        // Load which K rows are needed into thread-local registers
        for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
            const int kv = kv_start + kv_offset;
            is_needed_reg[kv_offset] = is_needed[kv];
        }

        // Load Q tile for the first head block
        for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q;
            if (q_idx < seq) {
                const T* Q_base = Q + HEAD_SIZE * (q_idx + seq * batch_idx);
                for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                    Q_tile[buf_idx][h][q] = Q_base[h];
                }
            }
        }

        // Load K tile for the first head block
        for (int kv = kv_start; kv < kv_start + THREAD_KV_BLOCK; ++kv) {
            const Index kv_idx = kv_tile_idx + kv;
            const int kv_offset = kv - kv_start;
            if (kv_idx < seq && is_needed_reg[kv_offset]) {
                const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx);
                for (int h = threadIdx.x; h < HEAD_BLOCK; h += blockDim.x) {
                    KV_tile[buf_idx][h][kv] = K_base[h];
                }
            }
        }

        // Wait for all threads to load the first K and Q tiles
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {

            // Prefetch next Q and K tile if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE) {
                // Load next Q tile
                int next_buf_idx = 1 - buf_idx;
                for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                    const Index q_idx = q_tile_idx * Q_BLOCK + q;
                    if (q_idx < seq) {
                        const T* Q_base = Q + HEAD_SIZE * (q_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                        for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                            Q_tile[next_buf_idx][h][q] = Q_base[h];
                        }
                    }
                }

                // Load next K tile
                for (int kv = kv_start; kv < kv_start + THREAD_KV_BLOCK; ++kv) {
                    const Index kv_idx = kv_tile_idx + kv;
                    const int kv_offset = kv - kv_start;
                    if (kv_idx < seq && is_needed_reg[kv_offset]) {
                        const T* K_base = K + HEAD_SIZE * (kv_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                        for (int h = threadIdx.x; h < HEAD_BLOCK; h += blockDim.x) {
                            KV_tile[next_buf_idx][h][kv] = K_base[h];
                        }
                    }
                }
            }
            // If this is the last head block, prefetch the first V tile
            else
            {
                // Load the first V tile
                int next_buf_idx = 1 - buf_idx;
                for (int kv = kv_start; kv < kv_start + THREAD_KV_BLOCK; ++kv) {
                    const Index kv_idx = kv_tile_idx + kv;
                    const int kv_offset = kv - kv_start;
                    if (kv_idx < seq && is_needed_reg[kv_offset]) {
                        const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx);
                        for (int h = threadIdx.x; h < HEAD_BLOCK; h += blockDim.x) {
                            KV_tile[next_buf_idx][h][kv] = V_base[h];
                        }
                    }
                }
            }

            // Compute K'Q directly into thread-local registers
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
                const int q = q_start + q_offset;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;

                if (q_idx < seq) {
                    for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
                        const int kv = kv_start + kv_offset;
                        const Index kv_idx = kv_tile_idx + kv;

                        // Skip computation if the position is masked
                        if (!std::isfinite(softmax_reg[q_offset][kv_offset]) || kv_idx >= seq) {
                            continue;
                        }

                        // Compute dot product for this position
                        Y dot_prod = 0;
                        for (int h = 0; h < HEAD_BLOCK; ++h) {
                            dot_prod += Y{Q_tile[buf_idx][h][q]} * Y{KV_tile[buf_idx][h][kv]};
                        }

                        // Scale and accumulate to register
                        softmax_reg[q_offset][kv_offset] += Y{scale} * dot_prod;
                    }
                }
            }
            __syncthreads();

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
        }

        // Apply softmax to thread-local registers
        for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
            const Index q_idx = q_tile_idx * Q_BLOCK + q_start + q_offset;
            if (q_idx < seq) {
                // Get pre-computed max and sumexp from maxsumexp
                const Index maxsumexp_idx = 2 * (q_idx + seq * batch_idx);
                const Y max_val = Y{maxsumexp[maxsumexp_idx]};
                const Y sumexp = Y{maxsumexp[maxsumexp_idx + 1]};

                // Apply softmax to each element in this row
                for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
                    const Index kv_idx = kv_tile_idx + kv_start + kv_offset;
                    if (std::isfinite(softmax_reg[q_offset][kv_offset]) && kv_idx < seq) {
                        softmax_reg[q_offset][kv_offset] =
                            std::exp(softmax_reg[q_offset][kv_offset] - max_val) / sumexp;
                    } else {
                        softmax_reg[q_offset][kv_offset] = 0;
                    }
                }
            } else {
                // Zero out rows beyond sequence length
                for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
                    softmax_reg[q_offset][kv_offset] = 0;
                }
            }
        }

        // Process head dimension in blocks to compute V @ softmax
        buf_idx = 0; // Reset buffer index
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK) {
            // Initialize the accumulator for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                    accum[q][h] = 0;
                }
            }
            __syncthreads();

            // Prefetch next V tile if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE) {
                int next_buf_idx = 1 - buf_idx;
                for (int kv = kv_start; kv < kv_start + THREAD_KV_BLOCK; ++kv) {
                    const Index kv_idx = kv_tile_idx + kv;
                    const int kv_offset = kv - kv_start;
                    if (kv_idx < seq && is_needed_reg[kv_offset]) {
                        const T* V_base = V + HEAD_SIZE * (kv_idx + seq * batch_idx) + (head_offset + HEAD_BLOCK);
                        for (int h = threadIdx.x; h < HEAD_BLOCK; h += blockDim.x) {
                            KV_tile[next_buf_idx][h][kv] = V_base[h];
                        }
                    }
                }
            }

            // Compute local sums for V @ softmax
            Y local_sums[THREAD_Q_BLOCK][HEAD_BLOCK];
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
                for (int h = 0; h < HEAD_BLOCK; ++h) {
                    local_sums[q_offset][h] = 0;
                }
            }

            // Compute V @ softmax using thread-local registers
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
                const int q = q_start + q_offset;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;

                if (q_idx < seq) {
                    for (int kv_offset = 0; kv_offset < THREAD_KV_BLOCK; ++kv_offset) {
                        const int kv = kv_start + kv_offset;
                        const Index kv_idx = kv_tile_idx + kv;

                        if (kv_idx < seq && softmax_reg[q_offset][kv_offset] > 0) {
                            const Y softmax_val = softmax_reg[q_offset][kv_offset];
                            for (int h = 0; h < HEAD_BLOCK; ++h) {
                                local_sums[q_offset][h] += softmax_val * Y{KV_tile[buf_idx][h][kv]};
                            }
                        }
                    }
                }
            }

            // Atomically add local sums to shared memory accumulator
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset) {
                const int q = q_start + q_offset;
                const Index q_idx = q_tile_idx * Q_BLOCK + q;

                if (q_idx < seq) {
                    for (int h = 0; h < HEAD_BLOCK; ++h) {
                        if (local_sums[q_offset][h] != 0) {
                            atomicAdd(&accum[q][h], local_sums[q_offset][h]);
                        }
                    }
                }
            }
            __syncthreads();

            // Atomic accumulation to global memory for this head block
            for (int q = threadIdx.x; q < Q_BLOCK; q += blockDim.x) {
                const Index q_idx = q_tile_idx * Q_BLOCK + q;
                if (q_idx < seq) {
                    T* A_base = A + HEAD_SIZE * (q_idx + seq * batch_idx) + head_offset;
                    for (int h = threadIdx.y; h < HEAD_BLOCK; h += blockDim.y) {
                        if (accum[q][h] != 0) {
                            if constexpr (std::is_same_v<T, fp32_t>) {
                                atomicAdd((float *)&A_base[h], accum[q][h]);
                            } else if constexpr (std::is_same_v<T, fp64_t>) {
                                atomicAdd((double *)&A_base[h], accum[q][h]);
                            } else {
                                // For other types, use a critical section or other atomic approach
                                // This is a simplified version that may not work for all types
                                A_base[h] = T(Y(A_base[h]) + accum[q][h]);
                            }
                        }
                    }
                }
            }
            __syncthreads();

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
