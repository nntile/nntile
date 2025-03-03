/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_maxsumexp/cuda.cu
 * CUDA kernel to compute maxsumexp((QK')/sqrt(d)) with masking
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/flash_maxsumexp/cuda.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mma.h>  // Include WMMA header

namespace nntile::kernel::flash_maxsumexp
{

template<typename T, typename Y, Index SEQ_BLOCK_SIZE_K, Index SEQ_BLOCK_SIZE_Q, Index HEAD_SIZE>
static __device__ void gemm_smem_sync(const T smem_K[SEQ_BLOCK_SIZE_K][HEAD_SIZE],
        const T smem_Q[SEQ_BLOCK_SIZE_Q][HEAD_SIZE],
        const bool_t smem_mask[SEQ_BLOCK_SIZE_Q][SEQ_BLOCK_SIZE_K],
        T scale,
        Y smem_KQ[SEQ_BLOCK_SIZE_Q][SEQ_BLOCK_SIZE_K])
{
    using namespace nvcuda::wmma;
    // Get warp and lane ids
    const Index thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Index warp_size = 32;
    const Index num_warps = blockDim.x * blockDim.y / warp_size;
    const Index warp_id = thread_id / warp_size;
    const Index lane_id = thread_id % warp_size;
    // Special case for nntile::fp32_fast_tf32_t using WMMA
    if constexpr (std::is_same<T, nntile::fp32_fast_tf32_t>::value)
    {
        auto smem_K_float = reinterpret_cast<const float *>(smem_K);
        auto smem_Q_float = reinterpret_cast<const float *>(smem_Q);
        // Define fragments using float precision TF32 supports only 16x16x8 grid
        fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
        fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
        fragment<accumulator, 16, 16, 8, float> c_frag;

        // Load data into fragments and perform matrix multiplication
        for(int w = warp_id; w < (SEQ_BLOCK_SIZE_K / 16) * (SEQ_BLOCK_SIZE_Q / 16); w += num_warps)
        {
            int i = (w / (SEQ_BLOCK_SIZE_K / 16)) * 16;
            int j = (w % (SEQ_BLOCK_SIZE_K / 16)) * 16;
            // Initialize accumulator fragment
            fill_fragment(c_frag, 0.0f);
            for(int k = 0; k < HEAD_SIZE; k += 8)
            {
                // Load matrix A and B from shared memory into fragments
                // K[SEQ_BLOCK_SIZE_K, HEAD_SIZE] - leading dimension is HEAD_SIZE
                load_matrix_sync(a_frag,
                    smem_K_float + j * HEAD_SIZE + k, HEAD_SIZE);
                // Q[SEQ_BLOCK_SIZE_Q, HEAD_SIZE] - leading dimension is HEAD_SIZE
                load_matrix_sync(b_frag,
                    smem_Q_float + i * HEAD_SIZE + k, HEAD_SIZE);
                // Perform matrix multiplication and accumulate
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            // Apply scale factor to the result
            for(int k = 0; k < c_frag.num_elements; ++k)
            {
                c_frag.x[k] *= float{scale};
            }
            // Store results back to shared memory
            store_matrix_sync(&smem_KQ[i][j], c_frag, SEQ_BLOCK_SIZE_K, mem_col_major);
        }
    }
    // General case
    else
    {
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE_Q; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE_K; j += blockDim.y)
            {
                Y sum = 0;
                for(int k = 0; k < HEAD_SIZE; ++k)
                {
                    // K[j,k] * Q[i,k] for K @ Q
                    sum += Y(smem_K[j][k]) * Y(smem_Q[i][k]);
                }
                smem_KQ[i][j] = sum * Y(scale);
            }
        }
    }
    __syncthreads();
}

template<typename T, Index SEQ_BLOCK_SIZE_K, Index SEQ_BLOCK_SIZE_Q, Index HEAD_SIZE>
__global__ void flash_maxsumexp_kernel(Index batch, Index seq,
        T scale, const T *K, const T *Q, const bool_t *mask, T *maxsumexp)
{
    using Scalar = typename T::repr_t;
    // Accumulator is fp64_t for fp64_t input type and fp32_t for other input types
    using ElementAccumulator = typename std::conditional<
        std::is_same<T, nntile::fp64_t>::value, double, float>::type;

    // Shared memory
    __shared__ T smem_Q[SEQ_BLOCK_SIZE_Q][HEAD_SIZE];
    __shared__ T smem_K[SEQ_BLOCK_SIZE_K][HEAD_SIZE];
    __shared__ bool_t smem_mask[SEQ_BLOCK_SIZE_Q][SEQ_BLOCK_SIZE_K];
    __shared__ ElementAccumulator smem_max[SEQ_BLOCK_SIZE_Q];
    __shared__ ElementAccumulator smem_sumexp[SEQ_BLOCK_SIZE_Q];
    __shared__ ElementAccumulator smem_KQ[SEQ_BLOCK_SIZE_Q][SEQ_BLOCK_SIZE_K];

    // Block indices - each block processes one tile of output
    const int block_col = blockIdx.x * SEQ_BLOCK_SIZE_Q;  // seq_j index
    const int batch_idx = blockIdx.y;                     // batch index

    // Get warp and lane ids
    const Index block_size = blockDim.x * blockDim.y;
    const Index thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Index warp_size = 32;
    const Index num_warps = block_size / warp_size;
    const Index warp_id = thread_id / warp_size;
    const Index lane_id = thread_id % warp_size;

    // Load maxsumexp[0:SEQ_BLOCK_SIZE_Q, batch_idx] into smem_max and smem_sumexp
    for(int j = thread_id; j < SEQ_BLOCK_SIZE_Q; j += block_size)
    {
        if(block_col + j < seq)
        {
            smem_max[j] = ElementAccumulator(maxsumexp[2 * (block_col + j + seq * batch_idx)]);
            smem_sumexp[j] = ElementAccumulator(maxsumexp[2 * (block_col + j + seq * batch_idx) + 1]);
            if(smem_sumexp[j] == 0)
            {
                smem_max[j] = -std::numeric_limits<ElementAccumulator>::infinity();
            }
        }
        else
        {
            smem_max[j] = -std::numeric_limits<ElementAccumulator>::infinity();
            smem_sumexp[j] = 0;
        }
    }

    // Load Q[block_row:block_row+SEQ_BLOCK_SIZE_Q, 0:HEAD_SIZE, batch_idx]
    for(int j = threadIdx.x; j < SEQ_BLOCK_SIZE_Q; j += blockDim.x)
    {
        if(block_col + j < seq)
        {
            for(int i = threadIdx.y; i < HEAD_SIZE; i += blockDim.y)
            {
                smem_Q[j][i] = Q[i + HEAD_SIZE * (block_col + j) + HEAD_SIZE * seq * batch_idx];
            }
        }
        else
        {
            for(int i = threadIdx.y; i < HEAD_SIZE; i += blockDim.y)
            {
                smem_Q[j][i] = 0;
            }
        }
    }

    // Loop through all blocks of K
    for(int block_row = 0; block_row < seq; block_row += SEQ_BLOCK_SIZE_K)
    {
        // Load K[block_row:block_row+SEQ_BLOCK_SIZE_K, 0:HEAD_SIZE, batch_idx]
        for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE_K; j += blockDim.y)
        {
            if(block_row + j < seq)
            {
                for(int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x)
                {
                    smem_K[j][i] = K[i + HEAD_SIZE * (block_row + j) + HEAD_SIZE * seq * batch_idx];
                }
            }
        }

        // Load mask[block_col:block_col+SEQ_BLOCK_SIZE_Q, block_row:block_row+SEQ_BLOCK_SIZE_K]
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE_Q; i += blockDim.x)
        {
            if(block_col + i < seq)
            {
                for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE_K; j += blockDim.y)
                {
                    if(block_row + j < seq)
                    {
                        smem_mask[i][j] = mask[block_row + j + seq * (block_col + i)];
                    }
                }
            }
        }

        // Synchronize threads within the block to ensure all data is loaded
        __syncthreads();

        // Compute tile product scale * K^T @ Q in Fortran order
        gemm_smem_sync<T, ElementAccumulator, SEQ_BLOCK_SIZE_K, SEQ_BLOCK_SIZE_Q, HEAD_SIZE>(smem_K,
                smem_Q, smem_mask, scale, smem_KQ);

        // Compute max in each column
        for(int j = thread_id; j < SEQ_BLOCK_SIZE_Q; j += block_size)
        {
            if(block_col + j < seq)
            {
                ElementAccumulator max_val = smem_max[j];
                ElementAccumulator sumexp = smem_sumexp[j];
                ElementAccumulator old_max = max_val;
                // Acquire maximum value in the column
                for(int i = 0; i < SEQ_BLOCK_SIZE_K; ++i)
                {
                    if(block_row + i < seq && smem_mask[j][i])
                    {
                        max_val = ::max(max_val, smem_KQ[j][i]);
                    }
                }
                // Update sumexp if needed
                if(max_val > old_max)
                {
                    sumexp *= ::exp(old_max - max_val);
                }
                // Compute sumexp in the column
                for(int i = 0; i < SEQ_BLOCK_SIZE_K; ++i)
                {
                    if(block_row + i < seq && smem_mask[j][i])
                    {
                        sumexp += ::exp(smem_KQ[j][i] - max_val);
                    }
                }
                // Update max and sumexp
                smem_max[j] = max_val;
                smem_sumexp[j] = sumexp;
            }
        }
        // Sync is required because we are using smem_mask, that will be updated
        // in the next iteration of the loop
        __syncthreads();
    }
    // Here we are not using __syncthreads() because there is only one thread
    // that "owns" required data in smem_max and smem_sumexp. With such an
    // approach we do not need any smem_max and smem_sumexp shared arrays
    // Store result to maxsumexp
    for(int j = thread_id; j < SEQ_BLOCK_SIZE_Q; j += block_size)
    {
        if(block_col + j < seq)
        {
            maxsumexp[2 * (block_col + j + seq * batch_idx)] = T{smem_max[j]};
            maxsumexp[2 * (block_col + j + seq * batch_idx) + 1] = T{smem_sumexp[j]};
        }
    }
}

template<typename T> // TODO: support SPLIT_K
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
        const T *K, const T *Q, const bool_t *mask, T *maxsumexp) noexcept
{
    // Shape of a temporary array maxsumexp_tmp is (2, seq, SPLIT_K, batch)
    // Shapes of K and Q are (head, seq, batch)
    // We virtually split K into SPLIT_K parts and compute maxsumexp for each
    // part separately
    // For this we reshape K into (head, seq/SPLIT_K, SPLIT_K, batch) without
    // touching the data (this reshape is free)
    // Then we compute maxsumexp for each split separately
    // 1. Entire Q is multiplied on split of K
    //      we virtually allocate intermediate temporary array shared_mem_tmp
    //      of shape (seq/SPLIT_K, seq, SPLIT_K, batch) and multiply a split
    //      of K.T by Q
    //      shared_mem_tmp[split_seq_idx, seq_idx, split_idx, batch_idx]
    //      = scale * sum( K[:, split_seq_idx, split_idx, batch_idx] *
    //                     Q[:, seq_idx, batch_idx] )
    // 2. We compute max and sumexp for each gemm of split of K
    //      maxsumexp_tmp[0, seq_idx, split_idx, batch_idx] =
    //          max(shared_mem_tmp[:, seq_idx, split_idx, batch_idx])
    //      maxsumexp_tmp[1, seq_idx, split_idx, batch_idx] =
    //          sumexp(shared_mem_tmp[:, seq_idx, split_idx, batch_idx] -
    //                 maxsumexp_tmp[0, seq_idx, split_idx, batch_idx])
    // Such a scheme will be done in future
    constexpr Index SEQ_BLOCK_SIZE_Q = 16;
    constexpr Index SEQ_BLOCK_SIZE_K = 32;
    // Shape of output blocking is defined by SEQ_BLOCK_SIZE_Q
    dim3 blocks((seq + SEQ_BLOCK_SIZE_Q - 1) / SEQ_BLOCK_SIZE_Q, batch);
    dim3 threads(16, 16);  // 256 threads per block

    // Calculate scale factor 1/sqrt(head)
    using Y = typename T::repr_t;
    T scale = T{Y(1) / std::sqrt(Y(head))};

    // Launch kernel
    if(head == 64)
    {
        flash_maxsumexp_kernel<T, SEQ_BLOCK_SIZE_K, SEQ_BLOCK_SIZE_Q, 64>
            <<<blocks, threads, 0, stream>>>(batch, seq, scale, K, Q, mask, maxsumexp);
    }
    else
    {
        std::cerr << "Unsupported head size" << std::endl;
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
