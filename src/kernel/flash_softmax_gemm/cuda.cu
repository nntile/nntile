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
    const T_gmem* gmem_ptr,
    T_smem* smem_ptr,
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
        if (col_in < BLOCK_COLS && row_in + 3 < BLOCK_ROWS) {
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

template<typename T_gmem, typename T_smem, typename T_accum, Index HEAD_SIZE, Index HEAD_BLOCK, Index THREAD_HEAD_BLOCK, Index Q_BLOCK, Index THREAD_Q_BLOCK, Index KV_BLOCK, Index THREAD_KV_BLOCK, Index KV_SPLIT, Index NUM_WARPS>
__global__ void flash_softmax_gemm_kernel(
    Index batch, Index seq, T_accum scale,
    const T_gmem *K, const T_gmem *Q, const bool_t *mask, const T_gmem *maxsumexp,
    const T_gmem *V, T_gmem *A)
{
    using namespace std;

    // Get global indices
    const Index thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Index block_size = blockDim.x * blockDim.y;
    const Index batch_idx = blockIdx.y;
    const Index q_block_idx = blockIdx.x;
    const Index kv_split_idx = blockIdx.z;

    // Calculate tile ranges
    const Index num_kv_blocks = (seq + KV_BLOCK - 1) / KV_BLOCK;
    const Index kv_split_num_blocks = (num_kv_blocks + KV_SPLIT - 1) / KV_SPLIT;
    const Index kv_split_size = kv_split_num_blocks * KV_BLOCK;
    const Index kv_block_start = kv_split_idx * kv_split_size;
    const Index kv_block_end = ::min(kv_block_start + kv_split_size, seq);

    // Thread indices for accessing the softmax tile
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int q_start = tx * THREAD_Q_BLOCK;
    const int kv_start = ty * THREAD_KV_BLOCK;
    const int head_start = ty * THREAD_HEAD_BLOCK;

    // Constants for warp-level processing
    constexpr int WARP_SIZE = 32;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    // Calculate number of warps in the block
    const int num_warps = block_size / WARP_SIZE;

    // Define tile dimensions for each warp to process
    constexpr int WARP_TILE_Q = 8;    // Width of tile processed by a warp
    constexpr int WARP_TILE_KV = 4;   // Height of tile processed by a warp

    // Calculate thread's position within the warp's tile using interleaved pattern
    const int thread_q_idx = lane_id % WARP_TILE_Q;  // WARP_TILE_Q threads in q dimension
    const int thread_kv_idx = lane_id / WARP_TILE_Q; // WARP_TILE_KV threads in kv dimension

    // Calculate total number of tiles
    constexpr int total_q_tiles = Q_BLOCK / WARP_TILE_Q; // Q_BLOCK is a multiple of WARP_TILE_Q
    constexpr int total_kv_tiles = KV_BLOCK / WARP_TILE_KV; // KV_BLOCK is a multiple of WARP_TILE_KV
    constexpr int total_tiles = total_q_tiles * total_kv_tiles;
    constexpr int softmax_reg_size = total_tiles / NUM_WARPS;

    // Shared memory allocations
    __shared__ T_smem Q_tile[2][HEAD_BLOCK][Q_BLOCK+1];    // Double buffered Q tile
    __shared__ T_smem KV_tile[2][HEAD_BLOCK][KV_BLOCK+1];  // Double buffered KV tile
    __shared__ T_smem softmax_tile[KV_BLOCK][Q_BLOCK+1];
    __shared__ bool is_needed[KV_BLOCK];

    // Thread-local registers for softmax tile
    T_accum softmax_reg[softmax_reg_size];
    bool is_needed_reg[THREAD_KV_BLOCK];

    // Process K,V blocks
    for (Index kv_block_idx = kv_block_start; kv_block_idx < kv_block_end; kv_block_idx += KV_BLOCK)
    {
        // Initialize buffer index for double buffering
        int buf_idx = 0;

        // Clear is_needed flags
        for (int kv = thread_id; kv < KV_BLOCK; kv += block_size)
        {
            is_needed[kv] = false;
        }
        __syncthreads();

        // Initialize softmax registers with mask information
        // We do it the same way, as we will do gemm K'Q to ensure maximal register usage
        for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps)
        {
            // Convert linear tile index to 2D coordinates
            const int q_tile_idx = tile_idx % total_q_tiles;
            const int kv_tile_idx = tile_idx / total_q_tiles;

            // Calculate starting positions for this tile
            const int q_tile_start = q_tile_idx * WARP_TILE_Q;
            const int kv_tile_start = kv_tile_idx * WARP_TILE_KV;

            // Each thread processes single element
            const int q = q_tile_start + thread_q_idx;
            const int kv = kv_tile_start + thread_kv_idx;
            const int reg_idx = (tile_idx - warp_id) / NUM_WARPS;

            if (bool{mask[kv + kv_block_idx + (q + q_block_idx * Q_BLOCK) * seq]})
            {
                softmax_reg[reg_idx] = 0;
                if (!is_needed[kv])
                {
                    is_needed[kv] = true;
                }
            }
            else
            {
                softmax_reg[reg_idx] = -std::numeric_limits<T_accum>::infinity();
            }
        }

        // Sync to get is_needed flags
        __syncthreads();

        // Load Q tile for the first head block
        gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, Q_BLOCK>(
            Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx),
            &Q_tile[buf_idx][0][0],
            HEAD_SIZE,
            Q_BLOCK+1,
            thread_id,
            block_size
        );

        // Load K tile for the first head block
        gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, KV_BLOCK>(
            K + HEAD_SIZE * (kv_block_idx + seq * batch_idx),
            &KV_tile[buf_idx][0][0],
            HEAD_SIZE,
            KV_BLOCK+1,
            thread_id,
            block_size
        );

        // Wait for all threads to load the first K and Q tiles
        __syncthreads();

        // Process head dimension in blocks to compute K'Q
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK)
        {
            // Buffer index for next iteration
            int next_buf_idx = 1 - buf_idx;

            // Prefetch next Q and K tile if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE)
            {
                // Load next Q tile
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, Q_BLOCK>(
                    Q + HEAD_SIZE * (q_block_idx * Q_BLOCK + seq * batch_idx)
                        + (head_offset + HEAD_BLOCK),
                    &Q_tile[next_buf_idx][0][0],
                    HEAD_SIZE,
                    Q_BLOCK+1,
                    thread_id,
                    block_size
                );

                // Load next K tile
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, KV_BLOCK>(
                    K + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                        + (head_offset + HEAD_BLOCK),
                    &KV_tile[next_buf_idx][0][0],
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
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, KV_BLOCK>(
                    V + HEAD_SIZE * (kv_block_idx + seq * batch_idx),
                    &KV_tile[next_buf_idx][0][0],
                    HEAD_SIZE,
                    KV_BLOCK+1,
                    thread_id,
                    block_size
                );
            }

            // Optimized K'Q multiplication with interleaved thread access pattern
            // Accumulating in thread-local registers and only writing to shared memory at the end
            {
                // Process tiles in a round-robin fashion across warps
                for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps)
                {
                    // Convert linear tile index to 2D coordinates
                    const int q_tile_idx = tile_idx % total_q_tiles;
                    const int kv_tile_idx = tile_idx / total_q_tiles;

                    // Calculate starting positions for this tile
                    const int q_tile_start = q_tile_idx * WARP_TILE_Q;
                    const int kv_tile_start = kv_tile_idx * WARP_TILE_KV;

                    // Each thread processes single element
                    const int q = q_tile_start + thread_q_idx;
                    const int kv = kv_tile_start + thread_kv_idx;
                    const int reg_idx = (tile_idx - warp_id) / NUM_WARPS;

                    // Only compute if this position is valid (not masked out)
                    if (::isfinite(softmax_reg[reg_idx]))
                    {
                        // Compute dot product for this (q,kv) position
                        T_accum dot_product = 0;

                        // Process HEAD_BLOCK elements in chunks for better register usage
                        for (int h = 0; h < HEAD_BLOCK; h += 4) {
                            // Load 4 elements from K and Q into registers
                            T_accum k_reg[4], q_reg[4];

                            #pragma unroll
                            for (int h_offset = 0; h_offset < 4; ++h_offset) {
                                k_reg[h_offset] = T_accum{KV_tile[buf_idx][h + h_offset][kv]};
                                q_reg[h_offset] = T_accum{Q_tile[buf_idx][h + h_offset][q]};
                            }

                            // Compute partial dot product
                            #pragma unroll
                            for (int h_offset = 0; h_offset < 4; ++h_offset) {
                                dot_product += k_reg[h_offset] * q_reg[h_offset];
                            }
                        }

                        // Accumulate to thread-local registers
                        softmax_reg[reg_idx] += dot_product * scale;
                    }
                }
            }
            __syncthreads();

            // Swap buffers for next iteration
            buf_idx = 1 - buf_idx;
        }

        // Apply softmax to thread-local registers and write results to shared memory
        for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
            // Convert linear tile index to 2D coordinates
            const int q_tile_idx = tile_idx % total_q_tiles;
            const int kv_tile_idx = tile_idx / total_q_tiles;

            // Calculate starting positions for this tile
            const int q_tile_start = q_tile_idx * WARP_TILE_Q;
            const int kv_tile_start = kv_tile_idx * WARP_TILE_KV;

            // Each thread writes its accumulated result (single element) to shared memory
            const int q = q_tile_start + thread_q_idx;
            const int kv = kv_tile_start + thread_kv_idx;
            const int reg_idx = (tile_idx - warp_id) / NUM_WARPS;

            // Get pre-computed max and sumexp from maxsumexp
            const Index maxsumexp_idx = 2 * (q + q_block_idx * Q_BLOCK + seq * batch_idx);
            const T_accum max_val = T_accum{maxsumexp[maxsumexp_idx]};
            const T_accum sumexp = T_accum{maxsumexp[maxsumexp_idx + 1]};

            // Only update if this position is valid (not masked out)
            if (::isfinite(softmax_reg[reg_idx]))
            {
                softmax_reg[reg_idx] = ::exp(softmax_reg[reg_idx] - max_val) / sumexp;
            }
            else
            {
                softmax_reg[reg_idx] = 0;
            }
            softmax_tile[kv][q] = T_smem{softmax_reg[reg_idx]};
        }
        __syncthreads();

        // Process head dimension in blocks to compute V @ softmax
        for (int head_offset = 0; head_offset < HEAD_SIZE; head_offset += HEAD_BLOCK)
        {
            // Prefetch next V tile if not at the last iteration
            if (head_offset + HEAD_BLOCK < HEAD_SIZE)
            {
                int next_buf_idx = 1 - buf_idx;
                gmem_to_smem_transposed_vec4<T_gmem, T_smem, HEAD_BLOCK, KV_BLOCK>(
                    V + HEAD_SIZE * (kv_block_idx + seq * batch_idx)
                        + (head_offset + HEAD_BLOCK),
                    &KV_tile[next_buf_idx][0][0],
                    HEAD_SIZE,
                    KV_BLOCK+1,
                    thread_id,
                    block_size);
            }

            __syncthreads();

            // Compute local sums for V @ softmax'
            T_accum local_sums[THREAD_Q_BLOCK][THREAD_HEAD_BLOCK];
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset)
            {
                for (int h_offset = 0; h_offset < THREAD_HEAD_BLOCK; ++h_offset)
                {
                    local_sums[q_offset][h_offset] = T_accum{0};
                }
            }

            // Compute V @ softmax' using thread-local registers
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset)
            {
                const int q = q_start + q_offset;
                const Index q_idx = q_block_idx * Q_BLOCK + q;

                for (int kv = 0; kv < KV_BLOCK; ++kv)
                {
                    const T_smem softmax_val_smem = softmax_tile[kv][q];
                    if (softmax_val_smem > 0)
                    {
                        for (int h_offset = 0; h_offset < THREAD_HEAD_BLOCK; ++h_offset)
                        {
                            local_sums[q_offset][h_offset] +=
                                T_accum{softmax_val_smem * KV_tile[buf_idx][h_offset+head_start][kv]};
                        }
                    }
                }
            }

            __syncthreads();

            // Atomic accumulation to global memory for this head block
            for (int q_offset = 0; q_offset < THREAD_Q_BLOCK; ++q_offset)
            {
                const int q = q_start + q_offset;
                const Index q_idx = q_block_idx * Q_BLOCK + q;
                T_gmem* A_base = A + HEAD_SIZE * (q_idx + seq * batch_idx) + head_offset;
                for (int h_offset = 0; h_offset < THREAD_HEAD_BLOCK; ++h_offset) {
                    if (local_sums[q_offset][h_offset] != 0.0)
                    {
                        atomicAdd(&A_base[h_offset+head_start], T_gmem{local_sums[q_offset][h_offset]});
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
    constexpr int THREAD_Q = 8;
    constexpr int THREAD_KV = 8;
    constexpr int THREAD_Q_BLOCK = 4;
    constexpr int THREAD_KV_BLOCK = 4;
    constexpr int THREAD_HEAD_BLOCK = 2;
    constexpr int NUM_WARPS = 2;
    constexpr int Q_BLOCK = THREAD_Q * THREAD_Q_BLOCK;
    constexpr int KV_BLOCK = THREAD_KV * THREAD_KV_BLOCK;
    constexpr int KV_SPLIT = 1;  // Balance between parallelism and overhead

    dim3 threads(THREAD_Q, THREAD_KV);

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
        if constexpr (std::is_same_v<T, nntile::fp32_t>)
        {
            flash_softmax_gemm_kernel<float, float, float,
                    HEAD_SIZE, HEAD_BLOCK, THREAD_HEAD_BLOCK, Q_BLOCK,
                    THREAD_Q_BLOCK, KV_BLOCK, THREAD_KV_BLOCK, KV_SPLIT, NUM_WARPS>
                <<<blocks, threads, 0, stream>>>(batch, seq, scale.value,
                    reinterpret_cast<const float*>(K), reinterpret_cast<const float*>(Q), mask,
                    reinterpret_cast<const float*>(maxsumexp), reinterpret_cast<const float*>(V),
                    reinterpret_cast<float*>(A));
        }
        else
        {
            std::cerr << "Unsupported type: " << typeid(T).name() << std::endl;
        }
        // TODO: enable other types T later
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

    // Make sure these values are equal or properly aligned
    static_assert(Q_BLOCK % (THREAD_Q * THREAD_Q_BLOCK) == 0,
                  "Q_BLOCK must be divisible by THREAD_Q * THREAD_Q_BLOCK");
    static_assert(KV_BLOCK % (THREAD_KV * THREAD_KV_BLOCK) == 0,
                  "KV_BLOCK must be divisible by THREAD_KV * THREAD_KV_BLOCK");
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
