/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_maxsumexp/cuda.cu
 * CUDA kernel to compute maxsumexp(mask((QK')/sqrt(d)))
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/flash_maxsumexp/cuda.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace nntile::kernel::flash_maxsumexp
{

/**
 * @brief Copy 2D block from global to shared memory with transposition
 *
 * @tparam T_gmem Type of data in global memory
 * @tparam T_smem Type of data in shared memory
 * @tparam BLOCK_ROWS Number of rows in the input block (columns in output)
 * @tparam BLOCK_COLS Number of columns in the input block (rows in output)
 *
 * @param gmem_ptr Pointer to the start of the block in global memory
 * @param smem_ptr Pointer to the start of the block in shared memory
 * @param gmem_ld Leading dimension of the global memory matrix
 * @param smem_ld Leading dimension of the shared memory matrix
 * @param thread_id Linear thread ID within the block
 * @param block_size Total number of threads in the block
 */
template<typename T_gmem, typename T_smem,
         Index BLOCK_ROWS, Index BLOCK_COLS>
__device__ void gmem_to_smem_transposed(
    const T_gmem* gmem_ptr,
    T_smem* smem_ptr,
    const Index gmem_ld,
    const Index smem_ld,
    const Index thread_id,
    const Index block_size)
{
    // Total number of elements to copy
    constexpr Index TOTAL_ELEMENTS = BLOCK_ROWS * BLOCK_COLS;
    // Make sure total elements is a multiple of 32 (warp size)
    static_assert(TOTAL_ELEMENTS % 32 == 0, "Total elements must be a multiple of 32");

    // Number of elements each thread will copy
    const Index ELEMENTS_PER_THREAD = (TOTAL_ELEMENTS + block_size - 1) / block_size;

    // Each thread copies ELEMENTS_PER_THREAD elements in an interleaved pattern
    for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        // Calculate linear index for this thread's current element
        const Index linear_idx = thread_id + i * block_size;

        // Skip if beyond the total elements
        if (linear_idx >= TOTAL_ELEMENTS) {
            break;
        }

        // Convert linear index to 2D coordinates in the input matrix
        const Index row_in = linear_idx / BLOCK_COLS;
        const Index col_in = linear_idx % BLOCK_COLS;

        // Transpose: row_out = col_in, col_out = row_in
        if (row_in < BLOCK_ROWS && col_in < BLOCK_COLS) {
            // Read from global memory in row-major order
            const T_gmem val = gmem_ptr[row_in + col_in * gmem_ld];

            // Write to shared memory with transposition (col_in becomes row, row_in becomes col)
            smem_ptr[col_in + row_in * smem_ld] = T_smem{val};
        }
    }
}

/**
 * @brief Vectorized copy 2D block from global to shared memory with transposition
 *
 * @tparam T_gmem Type of data in global memory
 * @tparam T_smem Type of data in shared memory
 * @tparam BLOCK_ROWS Number of rows in the input block (columns in output)
 * @tparam BLOCK_COLS Number of columns in the input block (rows in output)
 *
 * @param gmem_ptr Pointer to the start of the block in global memory
 * @param smem_ptr Pointer to the start of the block in shared memory
 * @param gmem_ld Leading dimension of the global memory matrix
 * @param smem_ld Leading dimension of the shared memory matrix
 * @param thread_id Linear thread ID within the block
 * @param block_size Total number of threads in the block
 */
template<typename T_gmem, typename T_smem,
         Index BLOCK_ROWS, Index BLOCK_COLS>
__device__ void gmem_to_smem_transposed_vec4(
    const T_gmem* __restrict gmem_ptr,
    T_smem* __restrict smem_ptr,
    const Index gmem_ld,
    const Index smem_ld,
    const Index thread_id,
    const Index block_size)
{
    // Ensure block rows is a multiple of 4 for vectorized loads
    static_assert(BLOCK_ROWS % 4 == 0, "Block rows must be a multiple of 4 for vectorized loads");

    // Total number of vector elements to copy (each vector contains 4 elements)
    constexpr Index TOTAL_VEC_ELEMENTS = (BLOCK_ROWS * BLOCK_COLS) / 4;

    // Number of vector elements each thread will copy
    const Index VEC_ELEMENTS_PER_THREAD = (TOTAL_VEC_ELEMENTS + block_size - 1) / block_size;

    // Each thread copies VEC_ELEMENTS_PER_THREAD vector elements
    #pragma unroll
    for (Index i = 0; i < VEC_ELEMENTS_PER_THREAD; ++i) {
        // Calculate linear index for this thread's current vector element
        const Index linear_vec_idx = thread_id + i * block_size;

        // Skip if beyond the total vector elements
        if (linear_vec_idx >= TOTAL_VEC_ELEMENTS) {
            break;
        }

        // Each vector spans 4 rows in the same column
        const Index col_in = linear_vec_idx / (BLOCK_ROWS / 4);
        const Index row_vec = linear_vec_idx % (BLOCK_ROWS / 4);
        const Index row_in = row_vec * 4;

        // Only process if within bounds
        // if (col_in < BLOCK_COLS && row_in + 3 < BLOCK_ROWS)
        {
            // Use vectorized load for better memory bandwidth
            // Load 4 consecutive rows from the same column
            float4 vec_val;

            // Manual load of 4 consecutive rows (can't use direct float4 load due to non-contiguous memory)
            vec_val = *reinterpret_cast<const float4*>(&gmem_ptr[row_in + col_in * gmem_ld]);

            // Store with transposition - the column in input becomes row in output
            // Each of the 4 rows becomes a column in the transposed output
            smem_ptr[col_in + (row_in + 0) * smem_ld] = T_smem{vec_val.x};
            smem_ptr[col_in + (row_in + 1) * smem_ld] = T_smem{vec_val.y};
            smem_ptr[col_in + (row_in + 2) * smem_ld] = T_smem{vec_val.z};
            smem_ptr[col_in + (row_in + 3) * smem_ld] = T_smem{vec_val.w};
        }
    }
}

/**
 * @brief Vectorized copy 2D block from global to shared memory with transposition
 *
 * @tparam T_gmem Type of data in global memory
 * @tparam T_smem Type of data in shared memory
 * @tparam BLOCK_ROWS Number of rows in the input block (columns in output)
 * @tparam BLOCK_COLS Number of columns in the input block (rows in output)
 *
 * @param gmem_ptr Pointer to the start of the block in global memory
 * @param smem_ptr Pointer to the start of the block in shared memory
 * @param gmem_ld Leading dimension of the global memory matrix
 * @param smem_ld Leading dimension of the shared memory matrix
 * @param thread_id Linear thread ID within the block
 * @param block_size Total number of threads in the block
 */
template<typename T_gmem, typename T_smem,
         Index BLOCK_ROWS, Index BLOCK_COLS>
__device__ void gmem_to_smem_vec4(
    const T_gmem* __restrict gmem_ptr,
    T_smem* __restrict smem_ptr,
    const Index gmem_ld,
    const Index smem_ld,
    const Index thread_id,
    const Index block_size)
{
    // Ensure block rows is a multiple of 4 for vectorized loads
    static_assert(BLOCK_ROWS % 4 == 0, "Block rows must be a multiple of 4 for vectorized loads");

    // Total number of vector elements to copy (each vector contains 4 elements)
    constexpr Index TOTAL_VEC_ELEMENTS = (BLOCK_ROWS * BLOCK_COLS) / 4;

    // Number of vector elements each thread will copy
    const Index VEC_ELEMENTS_PER_THREAD = (TOTAL_VEC_ELEMENTS + block_size - 1) / block_size;

    // Each thread copies VEC_ELEMENTS_PER_THREAD vector elements
    #pragma unroll
    for (Index i = 0; i < VEC_ELEMENTS_PER_THREAD; ++i) {
        // Calculate linear index for this thread's current vector element
        const Index linear_vec_idx = thread_id + i * block_size;

        // Skip if beyond the total vector elements
        if (linear_vec_idx >= TOTAL_VEC_ELEMENTS) {
            break;
        }

        // Each vector spans 4 rows in the same column
        const Index col_in = linear_vec_idx / (BLOCK_ROWS / 4);
        const Index row_vec = linear_vec_idx % (BLOCK_ROWS / 4);
        const Index row_in = row_vec * 4;

        // Only process if within bounds
        // if (col_in < BLOCK_COLS && row_in + 3 < BLOCK_ROWS)
        {
            // Use vectorized load for better memory bandwidth
            // Load 4 consecutive rows from the same column
            float4 vec_val;

            // Manual load of 4 consecutive rows (can't use direct float4 load due to non-contiguous memory)
            vec_val = *reinterpret_cast<const float4*>(&gmem_ptr[row_in + col_in * gmem_ld]);

            // Store with transposition - the column in input becomes row in output
            // Each of the 4 rows becomes a column in the transposed output
            smem_ptr[col_in * smem_ld + row_in + 0] = T_smem{vec_val.x};
            smem_ptr[col_in * smem_ld + row_in + 1] = T_smem{vec_val.y};
            smem_ptr[col_in * smem_ld + row_in + 2] = T_smem{vec_val.z};
            smem_ptr[col_in * smem_ld + row_in + 3] = T_smem{vec_val.w};
        }
    }
}

template<typename T_gmem, typename T_smem, typename T_accum,
         Index HEAD_SIZE, Index Q_BLOCK, Index K_BLOCK,
         Index KQ_HEAD_BLOCK, Index KQ_Q_TILE, Index KQ_K_TILE,
         Index K_SPLIT, Index NUM_COMP_WARPS>
__global__ void flash_maxsumexp_kernel(
    Index batch, Index seq, T_accum scale,
    const T_gmem * __restrict K, const T_gmem * __restrict Q,
    const bool_t * __restrict mask, T_gmem * __restrict maxsumexp)
// Every block of warps computes a single (K_BLOCK x Q_BLOCK) block
// of K'Q. Such a block of K'Q is a matrix multiplication of shape
// (HEAD_SIZE x K_BLOCK)^T x (HEAD_SIZE x Q_BLOCK) -> (K_BLOCK x Q_BLOCK).
// It is computed as a sequence of matrix multiplications of shape
// (KQ_HEAD_BLOCK x K_BLOCK)^T x (KQ_HEAD_BLOCK x Q_BLOCK) ->
// (K_BLOCK x Q_BLOCK).
// Therefore, matrix multiplications are done on top of:
// - block K of shape (KQ_HEAD_BLOCK x K_BLOCK),
// - block Q of shape (KQ_HEAD_BLOCK x Q_BLOCK).
// Every warp computes KQ_K_TILE x KQ_Q_TILE tiles of a block of K'Q.
// After computing K'Q, we compute maxsumexp(mask(K'Q))
{
    using namespace std;

    // Get global indices
    const Index thread_id = threadIdx.x;
    const Index block_size = blockDim.x;
    const Index batch_idx = blockIdx.z;
    const Index q_block_idx = blockIdx.x;
    const Index k_split_idx = blockIdx.y;

    // Calculate tile ranges
    const Index num_k_blocks = (seq + K_BLOCK - 1) / K_BLOCK;
    const Index k_split_num_blocks = (num_k_blocks + K_SPLIT - 1) / K_SPLIT;
    const Index k_split_size = k_split_num_blocks * K_BLOCK;
    const Index k_block_start = k_split_idx * k_split_size;
    const Index k_block_end = ::min(k_block_start + k_split_size, seq);

    // Constants for warp-level processing
    constexpr int WARP_SIZE = 32;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    // Helper functions for indexing into the 1D arrays
    auto mask_idx = [&](int q, int k) -> int {
        return q * (K_BLOCK+4) + k;
    };

    auto Q_idx = [&](int h, int q) -> int {
        return h * (Q_BLOCK + 32/KQ_HEAD_BLOCK) + q;
    };

    auto K_idx = [&](int buf, int h, int k) -> int {
        return buf * KQ_HEAD_BLOCK * (K_BLOCK + 32/KQ_HEAD_BLOCK) + h * (K_BLOCK + 32/KQ_HEAD_BLOCK) + k;
    };

    auto max_idx = [&](int k_tile, int q) -> int {
        return k_tile * Q_BLOCK + q;
    };

    auto sumexp_idx = [&](int k_tile, int q) -> int {
        return k_tile * Q_BLOCK + q;
    };

    // Number of tiles of a block of softmax(mask(K'Q)) in each dimension
    constexpr int KQ_K_TILE_NUM = K_BLOCK / KQ_K_TILE;
    constexpr int KQ_Q_TILE_NUM = Q_BLOCK / KQ_Q_TILE;
    constexpr int KQ_TILE_NUM = KQ_K_TILE_NUM * KQ_Q_TILE_NUM;

    // Number of tiles of softmax(mask(K'Q)) per computing warp in a block
    constexpr int KQ_TILE_PER_WARP = KQ_TILE_NUM / NUM_COMP_WARPS;

    // if(thread_id == 0)
    // {
    //     printf("KQ_K_TILE_NUM: %d\n", KQ_K_TILE_NUM);
    //     printf("KQ_Q_TILE_NUM: %d\n", KQ_Q_TILE_NUM);
    //     printf("KQ_TILE_NUM: %d\n", KQ_TILE_NUM);
    //     printf("KQ_TILE_PER_WARP: %d\n", KQ_TILE_PER_WARP);
    // }

    // Computing warp for softmax(mask(K'Q)) tile is the following grid of threads
    constexpr int KQ_WARP_K_THREADS = 8;
    constexpr int KQ_WARP_Q_THREADS = 32 / KQ_WARP_K_THREADS;

    // Number of softmax(mask(K'Q)) tile elements per thread in a computing warp
    constexpr int KQ_TILE_K_PER_THREAD = KQ_K_TILE / KQ_WARP_K_THREADS;
    constexpr int KQ_TILE_Q_PER_THREAD = KQ_Q_TILE / KQ_WARP_Q_THREADS;

    // Dynamic shared memory allocation
    extern __shared__ char shared_mem[];

    // Calculate offsets for different shared memory arrays
    constexpr int MAX_BLOCK_SIZE = Q_BLOCK * sizeof(T_smem);
    constexpr int SUMEXP_BLOCK_SIZE = Q_BLOCK * sizeof(T_smem);
    constexpr int Q_BLOCK_SIZE = HEAD_SIZE * (Q_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(T_smem);
    constexpr int K_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (K_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(T_smem);
    constexpr int MAX_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(T_smem);
    constexpr int SUMEXP_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(T_smem);
    constexpr int SHARED_MEM_SIZE = Q_BLOCK_SIZE + K_BLOCK_SIZE + MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE;

    // Assign pointers to shared memory regions with proper offsets
    T_accum* max_reduce = reinterpret_cast<T_accum*>(shared_mem);
    T_accum* sumexp_reduce = reinterpret_cast<T_accum*>(shared_mem + MAX_REDUCE_SIZE);
    T_smem* Q_block = reinterpret_cast<T_smem*>(shared_mem + MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE);
    T_smem* K_block = reinterpret_cast<T_smem*>(shared_mem + MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE + Q_BLOCK_SIZE);
    bool* mask_block = reinterpret_cast<bool*>(shared_mem + MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE + Q_BLOCK_SIZE + K_BLOCK_SIZE);

    // Init max and sum of exponents for the entire block of threads
    if(thread_id < Q_BLOCK)
    {
        max_reduce[max_idx(0, thread_id)] = maxsumexp[2 * (thread_id + seq * batch_idx)];
        sumexp_reduce[sumexp_idx(0, thread_id)] = maxsumexp[2 * (thread_id + seq * batch_idx) + 1];
        if(sumexp_reduce[sumexp_idx(0, thread_id)] == 0.0)
        {
            max_reduce[max_idx(0, thread_id)] = -std::numeric_limits<T_accum>::infinity();
        }
        for(int i = 1; i < KQ_K_TILE_NUM; i++)
        {
            max_reduce[max_idx(i, thread_id)] = -std::numeric_limits<T_accum>::infinity();
            sumexp_reduce[sumexp_idx(i, thread_id)] = 0.0;
        }
    }

    // for(int i = thread_id/Q_BLOCK + 1; i < KQ_K_TILE_NUM; i += block_size/Q_BLOCK)
    // // for(int i = thread_id/Q_BLOCK; i < KQ_K_TILE_NUM; i += block_size/Q_BLOCK)
    // {
    //     max_reduce[max_idx(i, thread_id % Q_BLOCK)] = -std::numeric_limits<T_accum>::infinity();
    //     sumexp_reduce[sumexp_idx(i, thread_id % Q_BLOCK)] = 0.0;
    // }
    __syncthreads();

    // Process K blocks
    for (Index k_block_idx = k_block_start; k_block_idx < k_block_end;
            k_block_idx += K_BLOCK)
    {
        // Thread-local registers for mask(K'Q)
        T_accum kq_reg[KQ_TILE_PER_WARP][KQ_TILE_K_PER_THREAD][
            KQ_TILE_Q_PER_THREAD];
        // Stage 1: Compute mask(K'Q) on registers
        {
            // Initialize buffer index for double buffering
            int buf_idx = 0;

            // Initialize mask tile
            int j = thread_id % Q_BLOCK;
            if(warp_id < NUM_COMP_WARPS)
            #pragma unroll
            for (int i = 16 * (thread_id / Q_BLOCK); i < K_BLOCK;
                    i += 16 * (NUM_COMP_WARPS * 32 / Q_BLOCK))
            {
                float4 mask_val = *reinterpret_cast<const float4*>(
                    &mask[k_block_idx + i + (j + q_block_idx * Q_BLOCK) * seq]);
                bool *mask_val_bool = reinterpret_cast<bool*>(&mask_val);
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                {
                    mask_block[mask_idx(j, i+k)] = mask_val_bool[k];
                }
            }
            __syncthreads();

            // Initialize K'Q block on registers with mask information
            // We do it the same way as gemm K'Q to ensure maximal register usage
            if(warp_id < NUM_COMP_WARPS)
            {
                #pragma unroll
                for(int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                        ++tile_idx_loop)
                {
                    int tile_idx = warp_id + tile_idx_loop * NUM_COMP_WARPS;
                    int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                    int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                    int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                    int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                    int q_local = q_tile_idx * KQ_Q_TILE + thread_q_idx;
                    int k_local = k_tile_idx * KQ_K_TILE + thread_k_idx;
                    #pragma unroll
                    for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                    {
                        #pragma unroll
                        for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                        {
                            if (mask_block[mask_idx(q_local + KQ_WARP_Q_THREADS * j,
                                    k_local + KQ_WARP_K_THREADS * i)])
                            {
                                kq_reg[tile_idx_loop][i][j] = 0;
                            }
                            else
                            {
                                kq_reg[tile_idx_loop][i][j] =
                                    -std::numeric_limits<T_accum>::infinity();
                            }
                        }
                    }
                }
            }

            // Sync to ensure all threads have loaded the mask(K'Q) block
            __syncthreads();

            // Load the first Q block of shape KQ_HEAD_BLOCK x Q_BLOCK
            if(warp_id < NUM_COMP_WARPS)
            {
                if (k_block_idx == k_block_start)
                {
                    gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
                        Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx),
                        Q_block + Q_idx(0, 0),
                        HEAD_SIZE,
                        Q_BLOCK + 32/KQ_HEAD_BLOCK,
                        thread_id,
                        NUM_COMP_WARPS * 32
                    );
                }

                // Load the first K block of shape KQ_HEAD_BLOCK x K_BLOCK
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, K_BLOCK>(
                    K + HEAD_SIZE * (k_block_idx + seq * batch_idx),
                    K_block + K_idx(buf_idx, 0, 0),
                    HEAD_SIZE,
                    K_BLOCK + 32/KQ_HEAD_BLOCK,
                    thread_id,
                    NUM_COMP_WARPS * 32
                );
            }

            // Process head dimension in chunks to compute entire block of K'Q
            // #pragma unroll 1
            for (int head_offset = 0; head_offset < HEAD_SIZE;
                    head_offset += KQ_HEAD_BLOCK)
            {
                // Synchronize to ensure all threads have loaded the current K and Q blocks
                __syncthreads();

                // Buffer index for next iteration
                int next_buf_idx = 1 - buf_idx;

                // Load next Q and K blocks
                if (head_offset + KQ_HEAD_BLOCK < HEAD_SIZE and warp_id < NUM_COMP_WARPS)
                {
                    // Load next Q block of shape KQ_HEAD_BLOCK x Q_BLOCK
                    if (k_block_idx == k_block_start)
                    {
                        gmem_to_smem_transposed_vec4<
                            T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
                            Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx)
                                + (head_offset + KQ_HEAD_BLOCK),
                            Q_block + Q_idx(head_offset + KQ_HEAD_BLOCK, 0),
                            HEAD_SIZE,
                            Q_BLOCK + 32/KQ_HEAD_BLOCK,
                            thread_id,
                            NUM_COMP_WARPS * 32
                        );
                    }

                    // Load next K block of shape KQ_HEAD_BLOCK x K_BLOCK
                    gmem_to_smem_transposed_vec4<
                        T_gmem, T_smem, KQ_HEAD_BLOCK, K_BLOCK>(
                        K + HEAD_SIZE * (k_block_idx + seq * batch_idx)
                            + (head_offset + KQ_HEAD_BLOCK),
                        K_block + K_idx(next_buf_idx, 0, 0),
                        HEAD_SIZE,
                        K_BLOCK + 32/KQ_HEAD_BLOCK,
                        thread_id,
                        NUM_COMP_WARPS * 32
                    );
                }

                __syncthreads();

                // Accumulate block of K'Q
                if(warp_id < NUM_COMP_WARPS)
                {
                    #pragma unroll
                    for (int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                            ++tile_idx_loop)
                    {
                        int tile_idx = warp_id + tile_idx_loop * NUM_COMP_WARPS;
                        int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                        int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                        int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                        int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                        int q = q_tile_idx * KQ_Q_TILE + thread_q_idx;
                        int k = k_tile_idx * KQ_K_TILE + thread_k_idx;
                        float a_vals[KQ_TILE_K_PER_THREAD],
                            b_vals[KQ_TILE_Q_PER_THREAD];
                        #pragma unroll 8
                        for (int h = 0; h < KQ_HEAD_BLOCK; ++h)
                        {
                            // #pragma unroll
                            // for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                            // {
                            //     // Load from K_block (it is transposed)
                            //     a_vals[i] = K_block[K_idx(buf_idx, h,
                            //         k + KQ_WARP_K_THREADS * i)];
                            // }
                            #pragma unroll
                            for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                            {
                                // Load from Q_block
                                b_vals[j] = Q_block[Q_idx(head_offset + h,
                                    q + KQ_WARP_Q_THREADS * j)];
                            }
                            #pragma unroll
                            for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                            {
                                T_smem k_val = K_block[K_idx(buf_idx, h,
                                    k + KQ_WARP_K_THREADS * i)];
                                #pragma unroll
                                for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                                {
                                    kq_reg[tile_idx_loop][i][j] +=
                                        k_val * b_vals[j];
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                // Swap buffers for next iteration
                buf_idx = 1 - buf_idx;
            }
        } // End of stage 1

        __syncthreads();

        // // Stage 2: Compute per-block per-warp maximums and update global per-warp maximums
        // if(warp_id < NUM_COMP_WARPS && thread_id == 0)
        // {
        //     printf("Stage 2 - warp %d processing tiles:\n", warp_id);
        //     for(int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP; ++tile_idx_loop)
        //     {
        //         int tile_idx = warp_id + tile_idx_loop * NUM_COMP_WARPS;
        //         int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
        //         int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
        //         printf("  tile_idx_loop=%d: tile_idx=%d (k=%d,q=%d)\n",
        //                tile_idx_loop, tile_idx, k_tile_idx, q_tile_idx);

        //         // Print some values from this tile
        //         printf("    First few values in tile:\n");
        //         for(int i = 0; i < 2; ++i)
        //         {
        //             for(int j = 0; j < 2; ++j)
        //             {
        //                 printf("    kq_reg[%d][%d][%d] = %f\n",
        //                        tile_idx_loop, i, j, kq_reg[tile_idx_loop][i][j]);
        //             }
        //         }
        //     }
        // }

        if(warp_id < NUM_COMP_WARPS)
        {
            // Multiply by scale inplace and compute per-warp maximums for current K'Q block
            #pragma unroll
            for (int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                    ++tile_idx_loop)
            {
                int tile_idx = warp_id + tile_idx_loop * NUM_COMP_WARPS;
                int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                int q_local = q_tile_idx * KQ_Q_TILE + thread_q_idx;

                // For each column in the tile
                #pragma unroll
                for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                {
                    int q_idx = q_local + j * KQ_WARP_Q_THREADS;
                    // Find the maximum value across all rows handled by this thread
                    T_accum new_warp_max = -std::numeric_limits<T_accum>::infinity();

                    #pragma unroll
                    for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                    {
                        kq_reg[tile_idx_loop][i][j] *= scale;
                        new_warp_max = ::max(new_warp_max, kq_reg[tile_idx_loop][i][j]);
                    }

                    // Use warp shuffle to perform reduction within each warp
                    #pragma unroll
                    for (int offset = WARP_SIZE/2; offset > KQ_WARP_Q_THREADS/2; offset /= 2)
                    {
                        // Get the max value from that thread
                        T_accum other = __shfl_down_sync(0xffffffff, new_warp_max, offset);
                        // Update our max value
                        new_warp_max = ::max(new_warp_max, other);
                    }

                    // Step 2: Only the first thread in each column updates the per-warp maximum
                    if (thread_k_idx == 0 && std::isfinite(new_warp_max))
                    {
                        // Update the per-warp maximum and sum of exponentials by comparing with the current value
                        T_accum current_warp_max = T_accum{max_reduce[max_idx(k_tile_idx, q_idx)]};
                        if(new_warp_max > current_warp_max)
                        {
                            max_reduce[max_idx(k_tile_idx, q_idx)] = new_warp_max;
                            sumexp_reduce[sumexp_idx(k_tile_idx, q_idx)] *= ::exp(current_warp_max - new_warp_max);
                        }
                    }
                    __syncwarp();
                }
            }

            __syncthreads();

            // Now compute the sum of exponentials for each column using the per-warp maximums
            #pragma unroll
            for (int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                    ++tile_idx_loop)
            {
                int tile_idx = warp_id + tile_idx_loop * NUM_COMP_WARPS;
                int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                int q_local = q_tile_idx * KQ_Q_TILE + thread_q_idx;

                // For each column in the tile
                #pragma unroll
                for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                {
                    int q_idx = q_local + j * KQ_WARP_Q_THREADS;
                    // Get the per-warp maximum value for this column
                    T_accum warp_max = T_accum{max_reduce[max_idx(k_tile_idx, q_idx)]};

                    // Compute local sum of exponentials for this column
                    T_accum new_warp_sum_exp = 0.0;

                    // Sum exp(x - warp_max) for all elements in this column
                    #pragma unroll
                    for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                    {
                        if (std::isfinite(kq_reg[tile_idx_loop][i][j]))
                        {
                            new_warp_sum_exp += ::exp(kq_reg[tile_idx_loop][i][j] - warp_max);
                        }
                    }

                    // Perform logarithmic reduction using shuffle operations
                    #pragma unroll
                    for (int offset = WARP_SIZE/2; offset > KQ_WARP_Q_THREADS/2; offset /= 2)
                    {
                        // Get the sum value from that thread
                        T_accum other = __shfl_down_sync(0xffffffff, new_warp_sum_exp, offset);
                        // Add to our sum
                        new_warp_sum_exp += other;
                    }

                    // Only the first thread in each column updates the per-warp sum
                    if (thread_k_idx == 0 && new_warp_sum_exp > 0.0)
                    {
                        // Update the per-warp sum by adding the current value
                        T_accum current_warp_sum_exp = sumexp_reduce[sumexp_idx(k_tile_idx, q_idx)];
                        sumexp_reduce[sumexp_idx(k_tile_idx, q_idx)] = current_warp_sum_exp + new_warp_sum_exp;
                    }
                    __syncwarp();
                }
            }
        }
        else
        {
            __syncthreads();
        } // End of stage 2
        __syncthreads();
    }

    // Stage 3: Combine per-warp maximums and sums into per-block values and update global memory
    if (thread_id < Q_BLOCK)
    {
        __syncthreads();
        int q_idx = q_block_idx * Q_BLOCK + thread_id;

        // Convert per-warp maximums into per-block maximum
        T_accum block_max_val = -std::numeric_limits<T_accum>::infinity();
        #pragma unroll
        for (int k = 0; k < KQ_K_TILE_NUM; ++k)
        {
            block_max_val = ::max(block_max_val, max_reduce[max_idx(k, thread_id)]);
        }

        // Update global max and sumexp only if global max (including value read from the global memory) is not -inf
        if (std::isfinite(block_max_val))
        {
            // Compute sum of exponentials across all k_tile_idx values
            T_accum block_sum_exp = 0;
            #pragma unroll
            for (int k = 0; k < KQ_K_TILE_NUM; ++k)
            {
                block_sum_exp += sumexp_reduce[sumexp_idx(k, thread_id)]
                    * ::exp(max_reduce[max_idx(k, thread_id)] - block_max_val);
            }
            // Write the final max and sumexp values to global memory
            maxsumexp[2 * (q_idx + seq * batch_idx)] = T_gmem{block_max_val};
            maxsumexp[2 * (q_idx + seq * batch_idx) + 1] = T_gmem{block_sum_exp};
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename T> // TODO: support SPLIT_K
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
        const T *K, const T *Q, const bool_t *mask, T *maxsumexp) noexcept
{
    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Launch kernel based on head size
    if (head == 64)
    {
        constexpr int HEAD_SIZE = 64;

        // Define block and grid sizes
        constexpr int NUM_THREADS = 128;  // Total number of threads per block
        constexpr int NUM_WARPS = NUM_THREADS / 32; // Number of warps per block
        constexpr int NUM_COMP_WARPS = 4; // Number of compute warps per block

        // Ensure we have the right number of threads for the warps
        static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32 (warp size)");

        // K'Q matmul is done by blocks:
        // K is split into blocks of size KQ_HEAD_BLOCK x K_BLOCK
        // Q is split into blocks of size KQ_HEAD_BLOCK x Q_BLOCK
        // K'Q is split into blocks of size K_BLOCK x Q_BLOCK
        constexpr int Q_BLOCK = 64;
        static_assert(Q_BLOCK % 32 == 0, "Q_BLOCK must be a multiple of 32");
        static_assert(Q_BLOCK <= NUM_THREADS, "Q_BLOCK must be less than or equal to NUM_THREADS");
        constexpr int K_BLOCK = 128;
        constexpr int KQ_HEAD_BLOCK = 16;

        // Split K and V into KV_SPLIT parts, each part is processed by a different
        // CUDA block. This is done to balance between parallelism and overhead.
        constexpr int K_SPLIT = 1;

        // Calculate shared memory size
        constexpr int MASK_BLOCK_SIZE = Q_BLOCK * (K_BLOCK+4) * sizeof(bool);
        constexpr int Q_BLOCK_SIZE = HEAD_SIZE * (Q_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(float);
        constexpr int K_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (K_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(float);
        constexpr int KQ_Q_TILE = 32;
        constexpr int KQ_K_TILE = 64;
        constexpr int KQ_K_TILE_NUM = K_BLOCK / KQ_K_TILE;
        constexpr int MAX_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(float);
        constexpr int SUMEXP_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(float);
        constexpr int SHARED_MEM_SIZE = MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE + Q_BLOCK_SIZE
            + K_BLOCK_SIZE + MASK_BLOCK_SIZE;
        static_assert(K_BLOCK * Q_BLOCK >= KQ_Q_TILE * KQ_K_TILE * NUM_COMP_WARPS,
                "K_BLOCK * Q_BLOCK must be greater than KQ_Q_TILE * KQ_K_TILE "
                "* NUM_COMP_WARPS");

        // Use 1D thread blocks instead of 2D
        dim3 threads(NUM_THREADS);
        dim3 blocks((seq + Q_BLOCK - 1) / Q_BLOCK, K_SPLIT, batch);

        if constexpr (std::is_same_v<T, nntile::fp32_t>)
        {
            cudaFuncSetAttribute(
                flash_maxsumexp_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, K_SPLIT, NUM_COMP_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_SIZE);

            flash_maxsumexp_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, K_SPLIT, NUM_COMP_WARPS>
                <<<blocks, threads, SHARED_MEM_SIZE, stream>>>(batch, seq, scale.value,
                    reinterpret_cast<const float*>(K), reinterpret_cast<const float*>(Q), mask,
                    reinterpret_cast<float*>(maxsumexp));
            gpuErrchk( cudaPeekAtLastError() );
        }
        else
        {
            std::cerr << "Unsupported type: " << typeid(T).name() << std::endl;
        }
        // TODO: enable other types T later
    }
    else if (head == 128)
    {
        constexpr int HEAD_SIZE = 128;

        // Define block and grid sizes
        constexpr int NUM_THREADS = 128;  // Total number of threads per block
        constexpr int NUM_WARPS = NUM_THREADS / 32; // Number of warps per block
        constexpr int NUM_COMP_WARPS = 4; // Number of compute warps per block

        // Ensure we have the right number of threads for the warps
        static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32 (warp size)");

        // K'Q matmul is done by blocks:
        // K is split into blocks of size KQ_HEAD_BLOCK x K_BLOCK
        // Q is split into blocks of size KQ_HEAD_BLOCK x Q_BLOCK
        // K'Q is split into blocks of size K_BLOCK x Q_BLOCK
        constexpr int Q_BLOCK = 64;
        static_assert(Q_BLOCK % 32 == 0, "Q_BLOCK must be a multiple of 32");
        static_assert(Q_BLOCK <= NUM_THREADS, "Q_BLOCK must be less than or equal to NUM_THREADS");
        constexpr int K_BLOCK = 128;
        constexpr int KQ_HEAD_BLOCK = 16;

        // Split K and V into KV_SPLIT parts, each part is processed by a different
        // CUDA block. This is done to balance between parallelism and overhead.
        constexpr int K_SPLIT = 1;

        // Calculate shared memory size
        constexpr int MASK_BLOCK_SIZE = Q_BLOCK * (K_BLOCK+4) * sizeof(bool);
        constexpr int Q_BLOCK_SIZE = HEAD_SIZE * (Q_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(float);
        constexpr int K_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (K_BLOCK+32/KQ_HEAD_BLOCK) * sizeof(float);
        constexpr int KQ_Q_TILE = 32;
        constexpr int KQ_K_TILE = 64;
        constexpr int KQ_K_TILE_NUM = K_BLOCK / KQ_K_TILE;
        constexpr int MAX_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(float);
        constexpr int SUMEXP_REDUCE_SIZE = KQ_K_TILE_NUM * Q_BLOCK * sizeof(float);
        constexpr int SHARED_MEM_SIZE = std::max(Q_BLOCK_SIZE + MASK_BLOCK_SIZE,
                Q_BLOCK_SIZE + K_BLOCK_SIZE + MAX_REDUCE_SIZE + SUMEXP_REDUCE_SIZE);
        static_assert(K_BLOCK * Q_BLOCK >= KQ_Q_TILE * KQ_K_TILE * NUM_COMP_WARPS,
                "K_BLOCK * Q_BLOCK must be greater than KQ_Q_TILE * KQ_K_TILE "
                "* NUM_COMP_WARPS");

        // Use 1D thread blocks instead of 2D
        dim3 threads(NUM_THREADS);
        dim3 blocks((seq + Q_BLOCK - 1) / Q_BLOCK, K_SPLIT, batch);

        if constexpr (std::is_same_v<T, nntile::fp32_t>)
        {
            cudaFuncSetAttribute(
                flash_maxsumexp_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, K_SPLIT, NUM_COMP_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_SIZE);

            flash_maxsumexp_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, K_SPLIT, NUM_COMP_WARPS>
                <<<blocks, threads, SHARED_MEM_SIZE, stream>>>(batch, seq, scale.value,
                    reinterpret_cast<const float*>(K), reinterpret_cast<const float*>(Q), mask,
                    reinterpret_cast<float*>(maxsumexp));
            gpuErrchk( cudaPeekAtLastError() );
        }
        else
        {
            std::cerr << "Unsupported type: " << typeid(T).name() << std::endl;
        }
        // TODO: enable other types T later
    }
    // TODO: enable other heads later
    else
    {
        std::cerr << "Unsupported head size: " << head << std::endl;
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_t *K, const fp32_t *Q, const bool_t *mask,
        fp32_t *maxsumexp) noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp64_t *K, const fp64_t *Q, const bool_t *mask,
        fp64_t *maxsumexp) noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const bf16_t *K, const bf16_t *Q, const bool_t *mask,
        bf16_t *maxsumexp) noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_tf32_t *K, const fp32_fast_tf32_t *Q, const bool_t *mask,
        fp32_fast_tf32_t *maxsumexp) noexcept;


template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_fp16_t *K, const fp32_fast_fp16_t *Q, const bool_t *mask,
        fp32_fast_fp16_t *maxsumexp) noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_bf16_t *K, const fp32_fast_bf16_t *Q, const bool_t *mask,
        fp32_fast_bf16_t *maxsumexp) noexcept;

} // namespace nntile::kernel::flash_maxsumexp
