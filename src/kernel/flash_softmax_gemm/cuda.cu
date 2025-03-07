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
         Index VS_HEAD_BLOCK, Index V_BLOCK, Index VS_HEAD_TILE,
         Index VS_Q_TILE, Index KV_SPLIT, Index NUM_WARPS>
__global__ void flash_softmax_gemm_kernel(
    Index batch, Index seq, T_accum scale,
    const T_gmem * __restrict K, const T_gmem * __restrict Q,
    const bool_t * __restrict mask, const T_gmem * __restrict maxsumexp,
    const T_gmem * __restrict V, T_gmem * __restrict A)
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
// In addition to computing K'Q, we also compute softmax(mask(K'Q))
// We denote softmax(mask(K'Q)) as S.
//
// Every block of warps computes a HEAD_SIZE x Q_BLOCK block of VS.
// VS is a matrix multiplication of shape
// (HEAD_SIZE x K_BLOCK) x (K_BLOCK x Q_BLOCK) -> HEAD_SIZE x Q_BLOCK.
// Due to possible large head dimension, we split row dimension HEAD_SIZE
// into VS_HEAD_BLOCK blocks. Then, each multiplication
// (VS_HEAD_BLOCK x K_BLOCK) x (K_BLOCK x Q_BLOCK) -> VS_HEAD_BLOCK x Q_BLOCK
// is computed as a sequence of matrix multiplications of shape
// (VS_HEAD_BLOCK x V_BLOCK) x (V_BLOCK x Q_BLOCK) -> VS_HEAD_BLOCK x Q_BLOCK,
// where K_BLOCK is split into V_BLOCK blocks.
// Every warp computes VS_HEAD_TILE x VS_Q_TILE tiles of a block of VS.
{
    using namespace std;

    // Get global indices
    const Index thread_id = threadIdx.x;
    const Index block_size = blockDim.x;
    const Index batch_idx = blockIdx.y;
    const Index q_block_idx = blockIdx.x;
    const Index kv_split_idx = blockIdx.z;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + K_BLOCK - 1) / K_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * K_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    // Constants for warp-level processing
    constexpr int WARP_SIZE = 32;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    // Dynamic shared memory allocation
    extern __shared__ char shared_mem[];

    // Calculate offsets for different shared memory arrays
    constexpr int MAX_BLOCK_SIZE = Q_BLOCK * sizeof(T_smem);
    // constexpr int SUMEXP_BLOCK_SIZE = Q_BLOCK * sizeof(T_smem);
    constexpr int Q_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(T_smem);
    // constexpr int K_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (K_BLOCK+1) * sizeof(T_smem);
    constexpr int SOFTMAX_BLOCK_SIZE = K_BLOCK * (Q_BLOCK+8) * sizeof(T_smem);
    // constexpr int V_BLOCK_SIZE = 2 * VS_HEAD_BLOCK * (V_BLOCK+1) * sizeof(T_smem);
    // constexpr int A_BLOCK_SIZE = VS_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(T_smem);

    // Assign pointers to shared memory regions with proper offsets
    T_smem* max_block = reinterpret_cast<T_smem*>(shared_mem);
    T_smem* sumexp_block = reinterpret_cast<T_smem*>(shared_mem + MAX_BLOCK_SIZE);
    bool* mask_block = reinterpret_cast<bool*>(shared_mem);
    T_smem* Q_block = reinterpret_cast<T_smem*>(shared_mem);
    T_smem* K_block = reinterpret_cast<T_smem*>(shared_mem + Q_BLOCK_SIZE);
    T_smem* softmax_block = reinterpret_cast<T_smem*>(shared_mem);
    T_smem* V_block = reinterpret_cast<T_smem*>(shared_mem + SOFTMAX_BLOCK_SIZE);
    // T_smem* A_block = reinterpret_cast<T_smem*>(shared_mem + SOFTMAX_BLOCK_SIZE + V_BLOCK_SIZE);

    // Helper functions for indexing into the 1D arrays
    auto mask_idx = [&](int q, int k) -> int {
        return q * (K_BLOCK+4) + k;
    };

    auto Q_idx = [&](int buf, int h, int q) -> int {
        return buf * KQ_HEAD_BLOCK * (Q_BLOCK+1) + h * (Q_BLOCK+1) + q;
    };

    auto K_idx = [&](int buf, int h, int k) -> int {
        return buf * KQ_HEAD_BLOCK * (K_BLOCK+1) + h * (K_BLOCK+1) + k;
    };

    auto softmax_idx = [&](int k, int q) -> int {
        return k * (Q_BLOCK+4) + q;
    };

    auto V_idx = [&](int buf, int h, int k) -> int {
        return buf * VS_HEAD_BLOCK * (V_BLOCK+1) + h * (V_BLOCK+1) + k;
    };

    auto A_idx = [&](int buf, int h, int q) -> int {
        return buf * VS_HEAD_BLOCK * (Q_BLOCK+1) + h * (Q_BLOCK+1) + q;
    };

    // Number of tiles of a block of softmax(mask(K'Q)) in each dimension
    constexpr int KQ_K_TILE_NUM = K_BLOCK / KQ_K_TILE;
    constexpr int KQ_Q_TILE_NUM = Q_BLOCK / KQ_Q_TILE;
    constexpr int KQ_TILE_NUM = KQ_K_TILE_NUM * KQ_Q_TILE_NUM;

    // Number of tiles of a block of VS in each dimension
    constexpr int VS_HEAD_TILE_NUM = VS_HEAD_BLOCK / VS_HEAD_TILE;
    constexpr int VS_Q_TILE_NUM = Q_BLOCK / VS_Q_TILE;
    constexpr int VS_TILE_NUM = VS_HEAD_TILE_NUM * VS_Q_TILE_NUM;

    // Number of tiles of softmax(mask(K'Q)) per warp in a block
    constexpr int KQ_TILE_PER_WARP = (KQ_TILE_NUM + NUM_WARPS - 1) / NUM_WARPS;

    // Number of tiles of V @ softmax per warp in a block
    constexpr int VS_TILE_PER_WARP = (VS_TILE_NUM + NUM_WARPS - 1) / NUM_WARPS;

    // Warp for softmax(mask(K'Q)) tile is the following grid of threads
    constexpr int KQ_WARP_K_THREADS = 4;
    constexpr int KQ_WARP_Q_THREADS = 32 / KQ_WARP_K_THREADS;

    // Warp for V @ softmax tile is the following grid of threads
    constexpr int VS_WARP_HEAD_THREADS = 8;
    constexpr int VS_WARP_Q_THREADS = 32 / VS_WARP_HEAD_THREADS;

    // Number of softmax(mask(K'Q)) tile elements per thread
    constexpr int KQ_TILE_K_PER_THREAD = KQ_K_TILE / KQ_WARP_K_THREADS;
    constexpr int KQ_TILE_Q_PER_THREAD = KQ_Q_TILE / KQ_WARP_Q_THREADS;

    // Number of V @ softmax tile elements per thread
    constexpr int VS_TILE_HEAD_PER_THREAD =
        VS_HEAD_TILE / VS_WARP_HEAD_THREADS;
    constexpr int VS_TILE_Q_PER_THREAD = VS_Q_TILE / VS_WARP_Q_THREADS;

    // Thread-local registers for max and sumexp of softmax(mask(K'Q))
    T_accum max_reg[KQ_Q_TILE_NUM][KQ_TILE_Q_PER_THREAD];
    T_accum sumexp_reg[KQ_Q_TILE_NUM][KQ_TILE_Q_PER_THREAD];

    for (int i = 2 * thread_id; i < Q_BLOCK; i += 2 * block_size)
    {
        float4 maxsumexp_val = *reinterpret_cast<const float4*>(
            &maxsumexp[2 * (i + q_block_idx * Q_BLOCK + seq * batch_idx)]);
        max_block[i] = maxsumexp_val.x;
        sumexp_block[i] = 1.0 / maxsumexp_val.y; // inverse of sumexp
        max_block[i+1] = maxsumexp_val.z;
        sumexp_block[i+1] = 1.0 / maxsumexp_val.w; // inverse of sumexp
    }
    __syncthreads();

    int q_lane_id = lane_id % KQ_WARP_Q_THREADS;
    #pragma unroll
    for (int i = 0; i < KQ_Q_TILE_NUM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
        {
            int q_idx = i * KQ_Q_TILE + q_lane_id + j * KQ_WARP_Q_THREADS;
            max_reg[i][j] = max_block[q_idx];
            sumexp_reg[i][j] = sumexp_block[q_idx];
        }
    }
    __syncthreads();

    // Process K,V blocks
    for (Index kv_block_idx = kv_block_start; kv_block_idx < kv_block_end;
            kv_block_idx += K_BLOCK)
    {
        // Stage 1: Compute softmax(mask(K'Q))
        {
            // Thread-local registers for softmax(mask(K'Q))
            T_accum softmax_reg[KQ_TILE_PER_WARP][KQ_TILE_K_PER_THREAD][
                KQ_TILE_Q_PER_THREAD];

            // Initialize buffer index for double buffering
            int buf_idx = 0;

            // Initialize mask tile
            int j = thread_id % Q_BLOCK;
            #pragma unroll
            for (int i = 16 * (thread_id / Q_BLOCK); i < K_BLOCK;
                    i += 16 * (block_size / Q_BLOCK))
            {
                float4 mask_val = *reinterpret_cast<const float4*>(
                    &mask[kv_block_idx + i + (j + q_block_idx * Q_BLOCK) * seq]);
                bool *mask_val_bool = reinterpret_cast<bool*>(&mask_val);
                for (int k = 0; k < 16; ++k)
                {
                    mask_block[mask_idx(j, i+k)] = T_smem(mask_val_bool[k]);
                }
            }
            __syncthreads();

            // Initialize K'Q block on registers with mask information
            // We do it the same way as gemm K'Q to ensure maximal register usage
            #pragma unroll
            for(int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                    ++tile_idx_loop)
            {
                int tile_idx = warp_id + tile_idx_loop * NUM_WARPS;
                if(tile_idx >= KQ_TILE_NUM)
                {
                    break;
                }
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
                            softmax_reg[tile_idx_loop][i][j] = 0;
                        }
                        else
                        {
                            softmax_reg[tile_idx_loop][i][j] =
                                -std::numeric_limits<T_accum>::infinity();
                        }
                    }
                }
            }
            __syncthreads();

            // Load the first Q block of shape KQ_HEAD_BLOCK x Q_BLOCK
            gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
                Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx),
                Q_block + Q_idx(buf_idx, 0, 0),
                HEAD_SIZE,
                Q_BLOCK + 1,
                thread_id,
                block_size
            );

            // Load the first K block of shape KQ_HEAD_BLOCK x K_BLOCK
            gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, K_BLOCK>(
                K + HEAD_SIZE * (kv_block_idx + seq * batch_idx),
                K_block + K_idx(buf_idx, 0, 0),
                HEAD_SIZE,
                K_BLOCK + 1,
                thread_id,
                block_size
            );

            // Wait for all threads to load the first K and Q blocks
            __syncthreads();

            // Process head dimension in chunks to compute entire block of K'Q
            #pragma unroll
            for (int head_offset = 0; head_offset < HEAD_SIZE;
                    head_offset += KQ_HEAD_BLOCK)
            {
                // Buffer index for next iteration
                int next_buf_idx = 1 - buf_idx;

                // Load next Q and K blocks
                if (head_offset + KQ_HEAD_BLOCK < HEAD_SIZE)
                {
                    // Load next Q block of shape KQ_HEAD_BLOCK x Q_BLOCK
                    gmem_to_smem_transposed_vec4<
                        T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
                        Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx)
                            + (head_offset + KQ_HEAD_BLOCK),
                        Q_block + Q_idx(next_buf_idx, 0, 0),
                        HEAD_SIZE,
                        Q_BLOCK + 1,
                        thread_id,
                        block_size
                    );

                    // Load next K block of shape KQ_HEAD_BLOCK x K_BLOCK
                    gmem_to_smem_transposed_vec4<
                        T_gmem, T_smem, KQ_HEAD_BLOCK, K_BLOCK>(
                        K + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                            + (head_offset + KQ_HEAD_BLOCK),
                        K_block + K_idx(next_buf_idx, 0, 0),
                        HEAD_SIZE,
                        K_BLOCK + 1,
                        thread_id,
                        block_size
                    );
                }

                // Accumulate block of K'Q
                #pragma unroll
                for (int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                        ++tile_idx_loop)
                {
                    int tile_idx = warp_id + tile_idx_loop * NUM_WARPS;
                    if(tile_idx >= KQ_TILE_NUM)
                    {
                        break;
                    }
                    int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                    int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                    int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                    int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                    int q = q_tile_idx * KQ_Q_TILE + thread_q_idx;
                    int k = k_tile_idx * KQ_K_TILE + thread_k_idx;
                    #pragma unroll
                    for (int h = 0; h < KQ_HEAD_BLOCK; ++h)
                    {
                        float a_vals[KQ_TILE_K_PER_THREAD],
                            b_vals[KQ_TILE_Q_PER_THREAD];
                        #pragma unroll
                        for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                        {
                            // Load from K_block (it is transposed)
                            a_vals[i] = K_block[K_idx(buf_idx, h,
                                k + KQ_WARP_K_THREADS * i)];
                        }
                        #pragma unroll
                        for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                        {
                            // Load from Q_block
                            b_vals[j] = Q_block[Q_idx(buf_idx, h,
                                q + KQ_WARP_Q_THREADS * j)];
                        }
                        #pragma unroll
                        for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                        {
                            #pragma unroll
                            for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                            {
                                softmax_reg[tile_idx_loop][i][j] +=
                                    a_vals[i] * b_vals[j];
                            }
                        }
                    }
                }

                __syncthreads();

                // Swap buffers for next iteration
                buf_idx = 1 - buf_idx;
            }

            // Apply softmax to thread-local registers and write results to shared memory
            #pragma unroll
            for (int tile_idx_loop = 0; tile_idx_loop < KQ_TILE_PER_WARP;
                    ++tile_idx_loop)
            {
                int tile_idx = warp_id + tile_idx_loop * NUM_WARPS;
                if(tile_idx >= KQ_TILE_NUM)
                {
                    break;
                }
                int q_tile_idx = (tile_idx % KQ_Q_TILE_NUM);
                int k_tile_idx = (tile_idx / KQ_Q_TILE_NUM);
                int thread_q_idx = lane_id % KQ_WARP_Q_THREADS;
                int thread_k_idx = lane_id / KQ_WARP_Q_THREADS;
                int q = q_tile_idx * KQ_Q_TILE + thread_q_idx;
                int k = k_tile_idx * KQ_K_TILE + thread_k_idx;
                #pragma unroll
                for (int j = 0; j < KQ_TILE_Q_PER_THREAD; ++j)
                {
                    const T_accum max_val = max_reg[q_tile_idx][j];
                    const T_accum sumexp = sumexp_reg[q_tile_idx][j];
                    #pragma unroll
                    for (int i = 0; i < KQ_TILE_K_PER_THREAD; ++i)
                    {
                        softmax_reg[tile_idx_loop][i][j] =
                            ::exp(scale * softmax_reg[tile_idx_loop][i][j]
                                - max_val
                            ) * sumexp;
                        softmax_block[softmax_idx(k + KQ_WARP_K_THREADS * i,
                            q + KQ_WARP_Q_THREADS * j)] =
                            T_smem{softmax_reg[tile_idx_loop][i][j]};
                    }
                }
            }
        } // End of stage 1

        // Stage 2: Compute VS
        {
            // Thread-local registers for the output VS
            T_accum A_reg[VS_TILE_PER_WARP][VS_TILE_HEAD_PER_THREAD][
                VS_TILE_Q_PER_THREAD];

            // Since VS is of shape HEAD_SIZE x Q_BLOCK, we process head
            // dimension in chunks of VS_HEAD_BLOCK. These chunks are
            // independent, they are processed sequentially one by one.
            #pragma unroll
            for (int head_offset = 0; head_offset < HEAD_SIZE;
                    head_offset += VS_HEAD_BLOCK)
            {
                int buf_idx = 0;

                // Load the first V block of shape VS_HEAD_BLOCK x V_BLOCK
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, VS_HEAD_BLOCK, V_BLOCK>(
                    V + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                        + head_offset,
                    V_block + V_idx(buf_idx, 0, 0),
                    HEAD_SIZE,
                    V_BLOCK + 1,
                    thread_id,
                    block_size);

                // Clear the output registers
                #pragma unroll
                for (int tile_idx_loop = 0; tile_idx_loop < VS_TILE_PER_WARP;
                        ++tile_idx_loop)
                {
                    #pragma unroll
                    for (int i = 0; i < VS_TILE_HEAD_PER_THREAD; ++i)
                    {
                        #pragma unroll
                        for (int j = 0; j < VS_TILE_Q_PER_THREAD; ++j)
                        {
                            A_reg[tile_idx_loop][i][j] = 0.0;
                        }
                    }
                }

                __syncthreads();

                // Here we process multiplication of V block of shape
                // VS_HEAD_BLOCK x K_BLOCK by S block of shape
                // K_BLOCK x Q_BLOCK. We do it in chunks of size V_BLOCK
                // along dimension of size K_BLOCK.
                #pragma unroll
                for (int v_block_idx = 0; v_block_idx < K_BLOCK;
                        v_block_idx += V_BLOCK)
                {
                    // Prefetch the next V block if not at the last iteration
                    if (v_block_idx + V_BLOCK < K_BLOCK)
                    {
                        int next_buf_idx = 1 - buf_idx;
                        int next_v_idx_start = v_block_idx + V_BLOCK
                            + kv_block_idx + seq * batch_idx;
                        gmem_to_smem_transposed_vec4<
                            T_gmem, T_smem, VS_HEAD_BLOCK, V_BLOCK>(
                            V + HEAD_SIZE * next_v_idx_start + head_offset,
                            V_block + V_idx(next_buf_idx, 0, 0),
                            HEAD_SIZE,
                            V_BLOCK + 1,
                            thread_id,
                            block_size);
                    }

                    // Process tiles in a round-robin fashion across warps
                    #pragma unroll
                    for (int tile_idx_loop = 0; tile_idx_loop < VS_TILE_PER_WARP;
                            ++tile_idx_loop)
                    {
                        int tile_idx = warp_id + tile_idx_loop * NUM_WARPS;
                        if(tile_idx >= VS_TILE_NUM)
                        {
                            break;
                        }
                        int head_tile_idx = (tile_idx % VS_HEAD_TILE_NUM);
                        int q_tile_idx = (tile_idx / VS_HEAD_TILE_NUM);
                        int thread_head_idx = lane_id % VS_WARP_HEAD_THREADS;
                        int thread_q_idx = lane_id / VS_WARP_HEAD_THREADS;
                        int h = head_tile_idx * VS_HEAD_TILE + thread_head_idx;
                        int q = q_tile_idx * VS_Q_TILE + thread_q_idx;
                        #pragma unroll
                        for (int v = 0; v < V_BLOCK; ++v)
                        {
                            float a_vals[VS_TILE_HEAD_PER_THREAD],
                                b_vals[VS_TILE_Q_PER_THREAD];
                            #pragma unroll
                            for (int i = 0; i < VS_TILE_HEAD_PER_THREAD; ++i)
                            {
                                // Load from V_block
                                a_vals[i] = V_block[V_idx(buf_idx,
                                        h + VS_WARP_HEAD_THREADS * i, v)];
                            }
                            #pragma unroll
                            for (int j = 0; j < VS_TILE_Q_PER_THREAD; ++j)
                            {
                                // Load from softmax_block
                                b_vals[j] = softmax_block[softmax_idx(
                                        v + v_block_idx,
                                        q + VS_WARP_Q_THREADS * j)];
                            }
                            #pragma unroll
                            for (int i = 0; i < VS_TILE_HEAD_PER_THREAD; ++i)
                            {
                                #pragma unroll
                                for (int j = 0; j < VS_TILE_Q_PER_THREAD; ++j)
                                {
                                    A_reg[tile_idx_loop][i][j] +=
                                        T_accum{a_vals[i] * b_vals[j]};
                                }
                            }
                        }
                    }

                    // Wait for all threads in the warp to finish before processing the next block
                    __syncthreads();

                    // Swap buffers for next iteration
                    buf_idx = 1 - buf_idx;
                }

                // Process tiles in a round-robin fashion across warps
                #pragma unroll
                for (int tile_idx_loop = 0; tile_idx_loop < VS_TILE_PER_WARP;
                        ++tile_idx_loop)
                {
                    int tile_idx = warp_id + tile_idx_loop * NUM_WARPS;
                    if(tile_idx >= VS_TILE_NUM)
                    {
                        break;
                    }
                    int head_tile_idx = (tile_idx % VS_HEAD_TILE_NUM);
                    int q_tile_idx = (tile_idx / VS_HEAD_TILE_NUM);
                    int thread_head_idx = lane_id % VS_WARP_HEAD_THREADS;
                    int thread_q_idx = lane_id / VS_WARP_HEAD_THREADS;
                    int h = head_tile_idx * VS_HEAD_TILE + thread_head_idx;
                    int q = q_tile_idx * VS_Q_TILE + thread_q_idx;
                    #pragma unroll
                    for (int i = 0; i < VS_TILE_HEAD_PER_THREAD; ++i)
                    {
                        #pragma unroll
                        for (int j = 0; j < VS_TILE_Q_PER_THREAD; ++j)
                        {
                            const Index head_idx_local = h + VS_WARP_HEAD_THREADS * i;
                            const Index q_idx_local = q + VS_WARP_Q_THREADS * j;
                            const Index head_idx = head_offset + head_idx_local;
                            const Index q_idx = q_block_idx * Q_BLOCK + q_idx_local;
                            const Index a_idx = head_idx + HEAD_SIZE * (q_idx + seq * batch_idx);
                            atomicAdd(&A[a_idx], T_gmem{A_reg[tile_idx_loop][i][j]});
                            //A_block[A_idx(buf_idx, head_idx_local, q_idx_local)] =
                            //    T_smem{output_reg[tile_idx_loop][i][j]};
                        }
                    }
                }

                __syncthreads();
            }
        }
    } // End of stage 2
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

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept
{
    // Define block and grid sizes
    constexpr int NUM_THREADS = 128;  // Total number of threads per block
    constexpr int NUM_WARPS = NUM_THREADS / 32; // Number of warps per block

    // K'Q matmul is done by blocks:
    // K is split into blocks of size KQ_HEAD_BLOCK x K_BLOCK
    // Q is split into blocks of size KQ_HEAD_BLOCK x Q_BLOCK
    // K'Q is split into blocks of size K_BLOCK x Q_BLOCK
    constexpr int Q_BLOCK = 64;
    constexpr int K_BLOCK = 64;
    constexpr int KQ_HEAD_BLOCK = 16;

    // V @ softmax is done by blocks:
    // V is split into blocks of size VS_HEAD_BLOCK x V_BLOCK
    // softmax is split into blocks of size V_BLOCK x Q_BLOCK
    // V @ softmax is split into blocks of size VS_HEAD_BLOCK x Q_BLOCK
    constexpr int V_BLOCK = 16;
    constexpr int VS_HEAD_BLOCK = 64;

    // Split K and V into KV_SPLIT parts, each part is processed by a different
    // CUDA block. This is done to balance between parallelism and overhead.
    constexpr int KV_SPLIT = 1;

    // Ensure we have the right number of threads for the warps
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32 (warp size)");

    // Use 1D thread blocks instead of 2D
    dim3 threads(NUM_THREADS);
    dim3 blocks((seq + Q_BLOCK - 1) / Q_BLOCK, batch, KV_SPLIT);

    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Clear the output
    cudaMemsetAsync(A, 0, batch * head * seq * sizeof(T), stream);

    // Launch kernel based on head size
    // Note: KQ_HEAD_BLOCK and VS_HEAD_BLOCK must be divisible by 4 for optimal vectorized memory access
    if (head == 64) {
        constexpr int HEAD_SIZE = 64;

        // Calculate shared memory size
        constexpr int Q_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(float);
        constexpr int K_BLOCK_SIZE = 2 * KQ_HEAD_BLOCK * (K_BLOCK+1) * sizeof(float);
        constexpr int SOFTMAX_BLOCK_SIZE = K_BLOCK * (Q_BLOCK+8) * sizeof(float);
        constexpr int V_BLOCK_SIZE = 2 * VS_HEAD_BLOCK * (V_BLOCK+1) * sizeof(float);
        // constexpr int A_BLOCK_SIZE = VS_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(float);
        constexpr int SHARED_MEM_SIZE = std::max(Q_BLOCK_SIZE + K_BLOCK_SIZE,
                SOFTMAX_BLOCK_SIZE + V_BLOCK_SIZE);

        constexpr int KQ_Q_TILE = 32;
        constexpr int KQ_K_TILE = 32;
        constexpr int VS_HEAD_TILE = 32;
        constexpr int VS_Q_TILE = 32;

        if constexpr (std::is_same_v<T, nntile::fp32_t>)
        {
            cudaFuncSetAttribute(
                flash_softmax_gemm_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, VS_HEAD_BLOCK, V_BLOCK, VS_HEAD_TILE,
                    VS_Q_TILE, KV_SPLIT, NUM_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 160000);

            flash_softmax_gemm_kernel<float, float, float,
                    HEAD_SIZE, Q_BLOCK, K_BLOCK, KQ_HEAD_BLOCK, KQ_Q_TILE,
                    KQ_K_TILE, VS_HEAD_BLOCK, V_BLOCK, VS_HEAD_TILE,
                    VS_Q_TILE, KV_SPLIT, NUM_WARPS>
                <<<blocks, threads, SHARED_MEM_SIZE, stream>>>(batch, seq, scale.value,
                    reinterpret_cast<const float*>(K), reinterpret_cast<const float*>(Q), mask,
                    reinterpret_cast<const float*>(maxsumexp), reinterpret_cast<const float*>(V),
                    reinterpret_cast<float*>(A));
            gpuErrchk( cudaPeekAtLastError() );
        }
        else
        {
            std::cerr << "Unsupported type: " << typeid(T).name() << std::endl;
        }
        // TODO: enable other types T later
    } // TODO: enable other heads later
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
