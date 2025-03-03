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
 * @brief Copy 2D block from global to shared memory
 *
 * @tparam T_gmem Type of data in global memory
 * @tparam T_smem Type of data in shared memory
 * @tparam BLOCK_ROWS Number of rows in the block
 * @tparam BLOCK_COLS Number of columns in the block
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
__device__ void gmem_to_smem(
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

        // Convert linear index to 2D coordinates
        const Index row = linear_idx / BLOCK_COLS;
        const Index col = linear_idx % BLOCK_COLS;

        // Copy element from global to shared memory
        if (row < BLOCK_ROWS && col < BLOCK_COLS) {
            smem_ptr[row + col * smem_ld] = T_smem{gmem_ptr[row + col * gmem_ld]};
        }
    }
}

/**
 * @brief Vectorized copy 2D block from global to shared memory
 *
 * @tparam T_gmem Type of data in global memory
 * @tparam T_smem Type of data in shared memory
 * @tparam BLOCK_ROWS Number of rows in the block
 * @tparam BLOCK_COLS Number of columns in the block
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
    const T_gmem* gmem_ptr,
    T_smem* smem_ptr,
    const Index gmem_ld,
    const Index smem_ld,
    const Index thread_id,
    const Index block_size)
{
    // Ensure block columns is a multiple of 4 for vectorized loads
    static_assert(BLOCK_COLS % 4 == 0, "Block columns must be a multiple of 4 for vectorized loads");

    // Total number of vector elements to copy (each vector contains 4 elements)
    constexpr Index TOTAL_VEC_ELEMENTS = (BLOCK_ROWS * BLOCK_COLS) / 4;

    // Number of vector elements each thread will copy
    constexpr Index VEC_ELEMENTS_PER_THREAD = (TOTAL_VEC_ELEMENTS + block_size - 1) / block_size;

    // Each thread copies VEC_ELEMENTS_PER_THREAD vector elements
    for (Index i = 0; i < VEC_ELEMENTS_PER_THREAD; ++i) {
        // Calculate linear index for this thread's current vector element
        const Index linear_vec_idx = thread_id + i * block_size;

        // Skip if beyond the total vector elements
        if (linear_vec_idx >= TOTAL_VEC_ELEMENTS) {
            break;
        }

        // Convert linear vector index to 2D coordinates
        // Each vector spans 4 columns
        const Index row = linear_vec_idx / (BLOCK_COLS / 4);
        const Index vec_col = linear_vec_idx % (BLOCK_COLS / 4);
        const Index col = vec_col * 4;

        // Copy vector element (4 consecutive values) from global to shared memory
        if (row < BLOCK_ROWS && col + 3 < BLOCK_COLS) {
            // Use vectorized load for better memory bandwidth
            const float4* gmem_vec_ptr = reinterpret_cast<const float4*>(&gmem_ptr[row + col * gmem_ld]);
            float4 vec_val = *gmem_vec_ptr;

            // Store individual elements to shared memory
            smem_ptr[row + col * smem_ld] = T_smem{reinterpret_cast<float*>(&vec_val)[0]};
            smem_ptr[row + (col + 1) * smem_ld] = T_smem{reinterpret_cast<float*>(&vec_val)[1]};
            smem_ptr[row + (col + 2) * smem_ld] = T_smem{reinterpret_cast<float*>(&vec_val)[2]};
            smem_ptr[row + (col + 3) * smem_ld] = T_smem{reinterpret_cast<float*>(&vec_val)[3]};
        }
    }
}

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
    constexpr Index BLOCK_SIZE = 512;
    // Ensure block rows is a multiple of 4 for vectorized loads
    static_assert(BLOCK_ROWS % 4 == 0, "Block rows must be a multiple of 4 for vectorized loads");

    // Total number of vector elements to copy (each vector contains 4 elements)
    constexpr Index TOTAL_VEC_ELEMENTS = (BLOCK_ROWS * BLOCK_COLS) / 4;

    // Number of vector elements each thread will copy
    constexpr Index VEC_ELEMENTS_PER_THREAD = (TOTAL_VEC_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Each thread copies VEC_ELEMENTS_PER_THREAD vector elements
    #pragma unroll
    for (Index i = 0; i < VEC_ELEMENTS_PER_THREAD; ++i) {
        // Calculate linear index for this thread's current vector element
        const Index linear_vec_idx = thread_id + i * BLOCK_SIZE;

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

template<typename T_gmem, typename T_smem, typename T_accum,
         Index HEAD_SIZE, Index KQ_HEAD_BLOCK, Index VS_HEAD_BLOCK,
         Index Q_BLOCK, Index KV_BLOCK, Index KV_SPLIT, Index NUM_WARPS>
__global__ void flash_softmax_gemm_kernel(
    Index batch, Index seq, T_accum scale,
    const T_gmem * __restrict K, const T_gmem * __restrict Q,
    const bool_t * __restrict mask, const T_gmem * __restrict maxsumexp,
    const T_gmem * __restrict V, T_gmem * __restrict A)
{
    using namespace std;

    // Get global indices
    const Index thread_id = threadIdx.x;
    const Index block_size = blockDim.x;
    const Index batch_idx = blockIdx.y;
    const Index q_block_idx = blockIdx.x;
    const Index kv_split_idx = blockIdx.z;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + KV_BLOCK - 1) / KV_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * KV_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    // Constants for warp-level processing
    constexpr int WARP_SIZE = 32;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    // Calculate number of warps in the block
    const int num_warps = block_size / WARP_SIZE;

    // Define tile dimensions for K'Q multiplication
    constexpr int WARP_TILE_KQ_Q = 8;    // Width of tile processed by a warp
    constexpr int WARP_TILE_KQ_K = 4;    // Height of tile processed by a warp

    // Calculate thread's position within the warp's tile using interleaved pattern
    const int thread_kq_q_idx = lane_id % WARP_TILE_KQ_Q;  // WARP_TILE_KQ_Q threads in q dimension
    const int thread_kq_k_idx = lane_id / WARP_TILE_KQ_Q;  // WARP_TILE_KQ_K threads in k dimension

    // Calculate total number of tiles
    constexpr int total_kq_q_tiles = Q_BLOCK / WARP_TILE_KQ_Q; // Q_BLOCK is a multiple of WARP_TILE_KQ_Q
    constexpr int total_kq_k_tiles = KV_BLOCK / WARP_TILE_KQ_K; // KV_BLOCK is a multiple of WARP_TILE_KQ_K
    constexpr int total_kq_tiles = total_kq_q_tiles * total_kq_k_tiles;
    constexpr int softmax_reg_size = total_kq_tiles / NUM_WARPS;

    // Define tile dimensions for VS' multiplication
    constexpr int WARP_TILE_VS_Q = 16;    // Width of tile processed by a warp
    constexpr int WARP_TILE_VS_H = 2;    // Height of tile processed by a warp (head dimension)

    // Calculate total number of tiles
    constexpr int total_vs_q_tiles = Q_BLOCK / WARP_TILE_VS_Q;
    constexpr int total_vs_h_tiles = VS_HEAD_BLOCK / WARP_TILE_VS_H;
    constexpr int total_vs_tiles = total_vs_q_tiles * total_vs_h_tiles;
    constexpr int output_reg_size = total_vs_tiles / NUM_WARPS;

    // Thread's position within the warp's tile using interleaved pattern
    const int thread_vs_q_idx = lane_id % WARP_TILE_VS_Q;  // 16 threads in q dimension
    const int thread_vs_h_idx = lane_id / WARP_TILE_VS_Q;  // 2 threads in h dimension

    // Dynamic shared memory allocation
    extern __shared__ char shared_mem[];

    // Calculate offsets for different shared memory arrays
    constexpr int Q_TILE_SIZE = 2 * KQ_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(T_smem);
    constexpr int K_TILE_SIZE = 2 * KQ_HEAD_BLOCK * (KV_BLOCK+1) * sizeof(T_smem);
    constexpr int SOFTMAX_TILE_SIZE = KV_BLOCK * (Q_BLOCK+1) * sizeof(T_smem);
    constexpr int V_TILE_SIZE = 2 * VS_HEAD_BLOCK * (KV_BLOCK+1) * sizeof(T_smem);
    constexpr int IS_NEEDED_SIZE = KV_BLOCK * sizeof(bool);
    constexpr int MAXSUMEXP_TILE_SIZE = 2 * Q_BLOCK * sizeof(T_smem);

    // This assertion is critical - it ensures that Q_tile and K_tile can fit within the space
    // that will later be reused for softmax_tile
    //static_assert(Q_TILE_SIZE + K_TILE_SIZE <= SOFTMAX_TILE_SIZE,
    //             "Q_tile and K_tile must fit in softmax_tile. Adjust Q_BLOCK or KV_BLOCK.");

    // Assign pointers to shared memory regions with proper offsets
    T_smem* Q_tile = reinterpret_cast<T_smem*>(shared_mem);
    T_smem* K_tile = reinterpret_cast<T_smem*>(shared_mem + Q_TILE_SIZE);
    T_smem* softmax_tile = reinterpret_cast<T_smem*>(shared_mem + Q_TILE_SIZE + K_TILE_SIZE); // Reuse the space of Q_tile and K_tile
    T_smem* V_tile = reinterpret_cast<T_smem*>(shared_mem + Q_TILE_SIZE + K_TILE_SIZE + SOFTMAX_TILE_SIZE);
    bool* is_needed = reinterpret_cast<bool*>(shared_mem + Q_TILE_SIZE + K_TILE_SIZE + SOFTMAX_TILE_SIZE + V_TILE_SIZE);
    T_smem* maxsumexp_tile = reinterpret_cast<T_smem*>(shared_mem + Q_TILE_SIZE + K_TILE_SIZE + SOFTMAX_TILE_SIZE + V_TILE_SIZE + KV_BLOCK * sizeof(bool));

    // Helper functions for indexing into the 1D arrays
    auto Q_idx = [&](int buf, int h, int q) -> int {
        return buf * KQ_HEAD_BLOCK * (Q_BLOCK+1) + h * (Q_BLOCK+1) + q;
    };

    auto K_idx = [&](int buf, int h, int k) -> int {
        return buf * KQ_HEAD_BLOCK * (KV_BLOCK+1) + h * (KV_BLOCK+1) + k;
    };

    auto softmax_idx = [&](int k, int q) -> int {
        return k * (Q_BLOCK+1) + q;
    };

    auto V_idx = [&](int buf, int h, int k) -> int {
        return buf * VS_HEAD_BLOCK * (KV_BLOCK+1) + h * (KV_BLOCK+1) + k;
    };

    // Thread-local registers for softmax tile
    constexpr int K_TILE = 16;
    constexpr int Q_TILE = 32;
    T_accum softmax_reg[KV_BLOCK / K_TILE][Q_BLOCK / Q_TILE][K_TILE / 4][Q_TILE / 8];
    T_smem maxsumexp_reg[Q_BLOCK][2];

    for (int i = 0; i < Q_BLOCK; ++i)
    {
        maxsumexp_reg[i][0] = 0.0;//maxsumexp[2 * (i + q_block_idx * Q_BLOCK + seq * batch_idx)];
        maxsumexp_reg[i][1] = 1.0;//maxsumexp[2 * (i + q_block_idx * Q_BLOCK + seq * batch_idx) + 1];
    }

    // Process K,V blocks
    for (Index kv_block_idx = kv_block_start; kv_block_idx < kv_block_end; kv_block_idx += KV_BLOCK)
    {
        // Initialize buffer index for double buffering
        int buf_idx = 0;

        // // Clear is_needed flags
        // for (int kv = thread_id; kv < KV_BLOCK; kv += block_size)
        // {
        //     is_needed[kv] = false;
        // }
        // __syncthreads();

        // Initialize softmax registers with mask information
        // We do it the same way, as we will do gemm K'Q to ensure maximal register usage
        #pragma unroll
        for(int tile_idx_loop = 0; tile_idx_loop < (KV_BLOCK / K_TILE) * (Q_BLOCK / Q_TILE); tile_idx_loop += NUM_WARPS)
        {
            int tile_idx = warp_id + tile_idx_loop;
            int q_tile_idx = (tile_idx % (Q_BLOCK / Q_TILE));
            int k_tile_idx = (tile_idx / (Q_BLOCK / Q_TILE));
            int thread_q_idx = lane_id % 8;
            int thread_k_idx = lane_id / 8;
            int q = q_block_idx * Q_BLOCK + q_tile_idx * Q_TILE + thread_q_idx;
            int k = kv_block_idx + k_tile_idx * K_TILE + thread_k_idx;
            #pragma unroll
            for (int i = 0; i < K_TILE / 4; ++i)
            {
                #pragma unroll
                for (int j = 0; j < Q_TILE / 8; ++j)
                {
                    if (k + 4 * i <= q + 8 * j)
                    {
                        softmax_reg[k_tile_idx][q_tile_idx][i][j] = 0;
                    }
                    else
                    {
                        softmax_reg[k_tile_idx][q_tile_idx][i][j] = -std::numeric_limits<T_accum>::infinity();
                    }
                }
            }
        }
        // for (int tile_idx = warp_id; tile_idx < total_kq_tiles; tile_idx += num_warps)
        // {
        //     // Convert linear tile index to 2D coordinates
        //     const int q_tile_idx = tile_idx % total_kq_q_tiles;
        //     const int k_tile_idx = tile_idx / total_kq_q_tiles;

        //     // Calculate starting positions for this tile
        //     const int q_tile_start = q_tile_idx * WARP_TILE_KQ_Q;
        //     const int k_tile_start = k_tile_idx * WARP_TILE_KQ_K;

        //     // Each thread processes single element
        //     const int q = q_tile_start + thread_kq_q_idx;
        //     const int k = k_tile_start + thread_kq_k_idx;
        //     const int reg_idx = (tile_idx - warp_id) / NUM_WARPS;

        //     if (kv_block_idx + k <= q + q_block_idx * Q_BLOCK) //bool{mask[k + kv_block_idx + (q + q_block_idx * Q_BLOCK) * seq]})
        //     {
        //         softmax_reg[reg_idx] = 0;
        //         // softmax_tile[softmax_idx(k, q)] = 0;
        //         // if (!is_needed[k])
        //         // {
        //         //     is_needed[k] = true;
        //         // }
        //     }
        //     else
        //     {
        //         // softmax_tile[softmax_idx(k, q)] = -std::numeric_limits<T_accum>::infinity();
        //         softmax_reg[reg_idx] = -std::numeric_limits<T_accum>::infinity();
        //     }
        // }

        // Sync to get is_needed flags
        // __syncthreads();

        // Load Q tile for the first head block
        gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
            Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx),
            Q_tile + Q_idx(buf_idx, 0, 0),
            HEAD_SIZE,
            Q_BLOCK+1,
            thread_id,
            block_size
        );

        // Load K tile for the first head block
        gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, KV_BLOCK>(
            K + HEAD_SIZE * (kv_block_idx + seq * batch_idx),
            K_tile + K_idx(buf_idx, 0, 0),
            HEAD_SIZE,
            KV_BLOCK+1,
            thread_id,
            block_size
        );

        // Wait for all threads to load the first K and Q tiles
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        #pragma unroll
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += KQ_HEAD_BLOCK)
        {
            // Buffer index for next iteration
            int next_buf_idx = 1 - buf_idx;

            // Prefetch next Q and K tile if not at the last iteration
            if (head_offset + KQ_HEAD_BLOCK < HEAD_SIZE)
            {
                // Load next Q tile
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, Q_BLOCK>(
                    Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx)
                        + (head_offset + KQ_HEAD_BLOCK),
                    Q_tile + Q_idx(next_buf_idx, 0, 0),
                    HEAD_SIZE,
                    Q_BLOCK+1,
                    thread_id,
                    block_size
                );

                // Load next K tile
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, KQ_HEAD_BLOCK, KV_BLOCK>(
                    K + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                        + (head_offset + KQ_HEAD_BLOCK),
                    K_tile + K_idx(next_buf_idx, 0, 0),
                    HEAD_SIZE,
                    KV_BLOCK+1,
                    thread_id,
                    block_size
                );
            }
            // If this is the last head block, prefetch the first V tile
            else
            {
                // Load the first V tile
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, VS_HEAD_BLOCK, KV_BLOCK>(
                    V + HEAD_SIZE * (kv_block_idx + seq * batch_idx),
                    V_tile + V_idx(next_buf_idx, 0, 0),
                    HEAD_SIZE,
                    KV_BLOCK+1,
                    thread_id,
                    block_size
                );
            }

            // Compute K'Q
            #pragma unroll
            for (int tile_idx_loop = 0; tile_idx_loop < (KV_BLOCK / K_TILE) * (Q_BLOCK / Q_TILE); tile_idx_loop += NUM_WARPS)
            {
                int tile_idx = warp_id + tile_idx_loop;
                int q_tile_idx = (tile_idx % (Q_BLOCK / Q_TILE));
                int k_tile_idx = (tile_idx / (Q_BLOCK / Q_TILE));
                int thread_q_idx = lane_id % 8;
                int thread_k_idx = lane_id / 8;
                int q = q_tile_idx * Q_TILE + thread_q_idx;
                int k = k_tile_idx * K_TILE + thread_k_idx;
                #pragma unroll
                for (int h = 0; h < KQ_HEAD_BLOCK; ++h)
                {
                    float a_vals[K_TILE / 4], b_vals[Q_TILE / 8];
                    #pragma unroll
                    for (int i = 0; i < K_TILE / 4; ++i)
                    {
                        // Load from K_tile - we want K[k+4*i][h] for the transpose
                        // Since data is stored in column-major format and was loaded transposed,
                        // we access it as K[h][k+4*i]
                        a_vals[i] = K_tile[K_idx(buf_idx, h, k + 4 * i)];
                    }
                    #pragma unroll
                    for (int j = 0; j < Q_TILE / 8; ++j)
                    {
                        // Load from Q_tile - we want Q[h][q+8*j]
                        // Data is already in the right format (column-major)
                        b_vals[j] = Q_tile[Q_idx(buf_idx, h, q + 8 * j)];
                    }
                    #pragma unroll
                    for (int i = 0; i < K_TILE / 4; ++i)
                    {
                        #pragma unroll
                        for (int j = 0; j < Q_TILE / 8; ++j)
                        {
                            // Computing (K^T * Q)_{k+4*i, q+8*j} = K[h][k+4*i] * Q[h][q+8*j]
                            softmax_reg[k_tile_idx][q_tile_idx][i][j] += a_vals[i] * b_vals[j];
                        }
                    }
                }
            }

            // // Optimized K'Q multiplication with interleaved thread access pattern
            // // Accumulating in thread-local registers and only writing to shared memory at the end
            // {
            //     // Process tiles in a round-robin fashion across warps
            //     for (int tile_idx = warp_id - NUM_WARPS / 2; tile_idx < total_kq_tiles; tile_idx += NUM_WARPS / 2)
            //     {
            //         // Convert linear tile index to 2D coordinates
            //         const int q_tile_idx = tile_idx % total_kq_q_tiles;
            //         const int k_tile_idx = tile_idx / total_kq_q_tiles;

            //         // Calculate starting positions for this tile
            //         const int q_tile_start = q_tile_idx * WARP_TILE_KQ_Q;
            //         const int k_tile_start = k_tile_idx * WARP_TILE_KQ_K;

            //         // Each thread processes single element
            //         const int q = q_tile_start + thread_kq_q_idx;
            //         const int k = k_tile_start + thread_kq_k_idx;
            //         const int reg_idx = (tile_idx - warp_id) / (NUM_WARPS / 2);

            //         //softmax_reg[reg_idx] = softmax_tile[softmax_idx(k, q)];

            //         // Only compute if this position is valid (not masked out)
            //         //if (::isfinite(softmax_reg[reg_idx]))
            //         //if (::isfinite(softmax_tile[softmax_idx(k, q)]))
            //         {
            //             // Compute dot product for this (q,kv) position
            //             T_accum dot_product = 0;

            //             // Process KQ_HEAD_BLOCK elements in chunks for better register usage
            //             #pragma unroll
            //             for (int h = 0; h < KQ_HEAD_BLOCK; h += 4) {
            //                 // Load 4 elements from K and Q into registers
            //                 T_accum k_reg[4], q_reg[4];

            //                 #pragma unroll
            //                 for (int h_offset = 0; h_offset < 4 && h + h_offset < KQ_HEAD_BLOCK; ++h_offset) {
            //                     k_reg[h_offset] = T_accum{K_tile[K_idx(buf_idx, h + h_offset, k)]};
            //                     q_reg[h_offset] = T_accum{Q_tile[Q_idx(buf_idx, h + h_offset, q)]};
            //                 }

            //                 // Compute partial dot product
            //                 #pragma unroll
            //                 for (int h_offset = 0; h_offset < 4 && h + h_offset < KQ_HEAD_BLOCK; ++h_offset) {
            //                     dot_product += k_reg[h_offset] * q_reg[h_offset];
            //                 }
            //             }

            //             // Accumulate to thread-local registers
            //             //softmax_reg[reg_idx] += dot_product * scale;
            //             softmax_tile[softmax_idx(k, q)] += dot_product;
            //         }
            //     }
            // }
            __syncthreads();

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
        }

        // Apply softmax to thread-local registers and write results to shared memory
        #pragma unroll
        for (int tile_idx_loop = 0; tile_idx_loop < (KV_BLOCK / K_TILE) * (Q_BLOCK / Q_TILE); tile_idx_loop += NUM_WARPS)
        {
            int tile_idx = warp_id + tile_idx_loop;
            int q_tile_idx = (tile_idx % (Q_BLOCK / Q_TILE));
            int k_tile_idx = (tile_idx / (Q_BLOCK / Q_TILE));
            int thread_q_idx = lane_id % 8;
            int thread_k_idx = lane_id / 8;
            int q = q_tile_idx * Q_TILE + thread_q_idx;
            int k = k_tile_idx * K_TILE + thread_k_idx;
            #pragma unroll
            for (int j = 0; j < Q_TILE / 8; ++j)
            {
                const T_accum max_val = maxsumexp_reg[q + 8 * j][0];
                const T_accum sumexp = maxsumexp_reg[q + 8 * j][1];
                #pragma unroll
                for (int i = 0; i < K_TILE / 4; ++i)
                {
                    softmax_reg[k_tile_idx][q_tile_idx][i][j] = ::exp(scale * softmax_reg[k_tile_idx][q_tile_idx][i][j] - max_val) / sumexp;
                    softmax_tile[softmax_idx(k + 4 * i, q + 8 * j)] = T_smem{softmax_reg[k_tile_idx][q_tile_idx][i][j]};
                }
            }
        }

        // for (int tile_idx = warp_id; tile_idx < total_kq_tiles; tile_idx += num_warps) {
        //     // Convert linear tile index to 2D coordinates
        //     const int q_tile_idx = tile_idx % total_kq_q_tiles;
        //     const int k_tile_idx = tile_idx / total_kq_q_tiles;

        //     // Calculate starting positions for this tile
        //     const int q_tile_start = q_tile_idx * WARP_TILE_KQ_Q;
        //     const int k_tile_start = k_tile_idx * WARP_TILE_KQ_K;

        //     // Each thread writes its accumulated result (single element) to shared memory
        //     const int q = q_tile_start + thread_kq_q_idx;
        //     const int k = k_tile_start + thread_kq_k_idx;
        //     const int reg_idx = (tile_idx - warp_id) / NUM_WARPS;

        //     // Get pre-computed max and sumexp from maxsumexp
        //     const Index maxsumexp_idx = 2 * (q + q_block_idx * Q_BLOCK + seq * batch_idx);
        //     const T_accum max_val = T_accum{maxsumexp[maxsumexp_idx]};
        //     const T_accum sumexp = T_accum{maxsumexp[maxsumexp_idx + 1]};

        //     // Only update if this position is valid (not masked out)
        //     softmax_reg[reg_idx] = scale * softmax_tile[softmax_idx(k, q)];
        //     if (::isfinite(softmax_reg[reg_idx]))
        //     {
        //         softmax_reg[reg_idx] = ::exp(softmax_reg[reg_idx] - max_val) / sumexp;
        //     }
        //     else
        //     {
        //         softmax_reg[reg_idx] = 0;
        //     }
        //     softmax_tile[softmax_idx(k, q)] = T_smem{softmax_reg[reg_idx]};
        // }
        __syncthreads();
        continue;

        // Process head dimension in blocks to compute V @ softmax
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += VS_HEAD_BLOCK)
        {
            //Prefetch next V tile if not at the last iteration
            if (head_offset + VS_HEAD_BLOCK < HEAD_SIZE)
            {
                int next_buf_idx = 1 - buf_idx;
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, VS_HEAD_BLOCK, KV_BLOCK>(
                    V + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                        + (head_offset + VS_HEAD_BLOCK),
                    V_tile + V_idx(next_buf_idx, 0, 0),
                    HEAD_SIZE,
                    KV_BLOCK+1,
                    thread_id,
                    block_size);
            }

            // Compute V @ softmax' using warp-based tiling approach
            {
                // Process tiles in a round-robin fashion across warps
                for (int tile_idx = warp_id; tile_idx < total_vs_tiles; tile_idx += num_warps) {
                    // Convert linear tile index to 2D coordinates
                    const int h_tile_idx = tile_idx % total_vs_h_tiles;
                    const int q_tile_idx = tile_idx / total_vs_h_tiles;

                    // Calculate starting positions for this tile
                    const int q_tile_start = q_tile_idx * WARP_TILE_VS_Q;
                    const int h_tile_start = h_tile_idx * WARP_TILE_VS_H;

                    // Each thread processes a single element in the output tile
                    const int q = q_tile_start + thread_vs_q_idx;
                    const int h = h_tile_start + thread_vs_h_idx;
                    const int reg_idx = (tile_idx - warp_id) / num_warps;

                    T_accum output_reg = 0;

                    // Only compute if within bounds
                    if (q < Q_BLOCK && h < VS_HEAD_BLOCK) {
                        // Compute dot product across all KV elements
                        for (int kv = 0; kv < KV_BLOCK; ++kv) {
                            // Only process if this KV element is needed
                            // if (is_needed[kv])
                            {
                                const T_smem softmax_val = softmax_tile[softmax_idx(kv, q)];

                                // Only accumulate if softmax value is non-zero and finite
                                if (softmax_val > 0 && ::isfinite(softmax_val)) {
                                    const T_smem v_val = V_tile[V_idx(buf_idx, h, kv)];
                                    if (::isfinite(v_val)) {
                                        output_reg += T_accum{softmax_val * v_val};
                                    }
                                }
                            }
                        }
                    }
                    const Index q_idx = q_block_idx * Q_BLOCK + q;
                    T_gmem* A_base = A + HEAD_SIZE * (q_idx + seq * batch_idx) + head_offset;

                    atomicAdd(&A_base[h], T_gmem{output_reg});
                }
            }

            __syncthreads();

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
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

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept
{
    // Define block and grid sizes
    constexpr int NUM_THREADS = 64;  // Total number of threads per block
    constexpr int NUM_WARPS = NUM_THREADS / 32;     // Number of warps per block
    constexpr int Q_BLOCK = 32;      // Size of Q block
    constexpr int KV_BLOCK = 32;     // Size of KV block
    constexpr int KV_SPLIT = 128;      // Balance between parallelism and overhead

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
        constexpr int KQ_HEAD_BLOCK = 16;  // Process in 2 blocks, must be divisible by 4
        constexpr int VS_HEAD_BLOCK = 16;  // Process in 2 blocks, must be divisible by 4

        // Calculate shared memory size
        constexpr int Q_TILE_SIZE = 2 * KQ_HEAD_BLOCK * (Q_BLOCK+1) * sizeof(float);
        constexpr int K_TILE_SIZE = 2 * KQ_HEAD_BLOCK * (KV_BLOCK+1) * sizeof(float);
        constexpr int SOFTMAX_TILE_SIZE = KV_BLOCK * (Q_BLOCK+1) * sizeof(float);
        constexpr int V_TILE_SIZE = 2 * VS_HEAD_BLOCK * (KV_BLOCK+1) * sizeof(float);
        constexpr int IS_NEEDED_SIZE = KV_BLOCK * sizeof(bool);
        constexpr int MAXSUMEXP_TILE_SIZE = 2 * Q_BLOCK * sizeof(float);
        constexpr int SHARED_MEM_SIZE = Q_TILE_SIZE + K_TILE_SIZE + SOFTMAX_TILE_SIZE + V_TILE_SIZE + IS_NEEDED_SIZE + MAXSUMEXP_TILE_SIZE;
        //static_assert(Q_TILE_SIZE + K_TILE_SIZE <= SOFTMAX_TILE_SIZE, "Q_tile and K_tile must fit in softmax_tile");

        // This assertion is critical - it ensures that Q_tile and K_tile can fit within the space
        // that will later be reused for softmax_tile
        //static_assert(Q_TILE_SIZE + K_TILE_SIZE <= SOFTMAX_TILE_SIZE,
        //             "Q_tile and K_tile must fit in softmax_tile. Adjust Q_BLOCK or KV_BLOCK.");

        if constexpr (std::is_same_v<T, nntile::fp32_t>)
        {
            cudaFuncSetAttribute(
                flash_softmax_gemm_kernel<float, float, float,
                    HEAD_SIZE, KQ_HEAD_BLOCK, VS_HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT, NUM_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 160000);

            flash_softmax_gemm_kernel<float, float, float,
                    HEAD_SIZE, KQ_HEAD_BLOCK, VS_HEAD_BLOCK, Q_BLOCK, KV_BLOCK, KV_SPLIT, NUM_WARPS>
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
