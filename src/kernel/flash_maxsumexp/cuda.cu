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
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/layout/matrix.h>

namespace nntile::kernel::flash_maxsumexp
{

template<typename T>
__global__ void flash_maxsumexp_kernel(Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask, T *maxsumexp)
{
    using Scalar = typename T::repr_t;
    using ElementAccumulator = typename std::conditional<
        std::is_same<T, fp64_t>::value, double, float>::type;

    // Define tile sizes
    constexpr int SEQ_BLOCK_SIZE = 32; // Sequence is processed in parallel
    constexpr int HEAD_SIZE = 64; // Each head is processed in a single block

    // Shared memory
    __shared__ T smem_Q[HEAD_SIZE][SEQ_BLOCK_SIZE];
    __shared__ T smem_K[HEAD_SIZE][SEQ_BLOCK_SIZE];
    __shared__ bool_t smem_mask[SEQ_BLOCK_SIZE][SEQ_BLOCK_SIZE];
    __shared__ ElementAccumulator smem_max[SEQ_BLOCK_SIZE];
    __shared__ ElementAccumulator smem_sumexp[SEQ_BLOCK_SIZE];
    __shared__ ElementAccumulator smem_QK[SEQ_BLOCK_SIZE][SEQ_BLOCK_SIZE];

    // Block indices - each block processes one tile of output
    const int block_col = blockIdx.x * SEQ_BLOCK_SIZE;  // seq_j index
    const int batch_idx = blockIdx.y;                   // batch index

    // Load maxsumexp[0:seq, batch_idx] into smem_max and smem_sumexp
    for(int j = threadIdx.x; j < SEQ_BLOCK_SIZE; j += blockDim.x)
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
    }
    __syncthreads();

    // Load Q[0:head, block_row:block_row+SEQ_BLOCK_SIZE, batch_idx]
    for(int i = threadIdx.x; i < head; i += blockDim.x)
    {
        for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
        {
            if(block_col + j < seq)
            {
                smem_Q[i][j] = Q[i + head * (block_col + j) + head * seq * batch_idx];
            }
        }
    }
    __syncthreads();

    for(int block_row = 0; block_row < seq; block_row += SEQ_BLOCK_SIZE)
    {
        // Clear accumulator
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
            {
                smem_QK[i][j] = ElementAccumulator(0);
            }
        }

        // Load K[0:head, block_col:block_col+SEQ_BLOCK_SIZE, batch_idx]
        for(int i = threadIdx.x; i < head; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
            {
                if(block_row + j < seq)
                {
                    smem_K[i][j] = K[i + head * (block_row + j) + head * seq * batch_idx];
                }
            }
        }

        // Load mask[block_row:block_row+SEQ_BLOCK_SIZE, block_col:block_col+SEQ_BLOCK_SIZE]
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
            {
                if(block_row + i < seq && block_col + j < seq)
                {
                    smem_mask[i][j] = mask[block_row + i + seq * (block_col + j)];
                }
            }
        }
        __syncthreads();

        // Compute tile product scale * K^T @ Q in Fortran order
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE; i += blockDim.x)
        {
            if(block_row + i < seq)  // Row of K^T
            {
                for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
                {
                    if(block_col + j < seq)  // Column of Q
                    {
                        ElementAccumulator sum = 0;
                        for(int k = 0; k < head; ++k)
                        {
                            // K[k,i] * Q[k,j] for K^T @ Q
                            sum += Scalar(smem_K[k][i]) * Scalar(smem_Q[k][j]);
                        }
                        smem_QK[i][j] = sum * ElementAccumulator(scale);
                    }
                }
            }
        }
        __syncthreads();

        // Compute max in each row
        for(int j = threadIdx.x; j < SEQ_BLOCK_SIZE; j += blockDim.x)
        {
            // Get maximum in a very simple way with a single thread per group
            if(block_col + j < seq && threadIdx.y == 0)
            {
                ElementAccumulator max_val = smem_max[j];
                ElementAccumulator old_max_val = max_val;
                for(int i = 0; i < SEQ_BLOCK_SIZE; ++i)
                {
                    if(block_row + i < seq && smem_mask[i][j])
                    {
                        max_val = ::max(max_val, smem_QK[i][j]);
                    }
                }
                if(max_val > old_max_val)
                {
                    smem_sumexp[j] *= ::exp(old_max_val - max_val);
                    smem_max[j] = max_val;
                }
            }
        }
        __syncthreads();

        // Compute sumexp in each row
        for(int j = threadIdx.x; j < SEQ_BLOCK_SIZE; j += blockDim.x)
        {
            // Get sumexp in a very simple way with a single thread per group
            if(block_col + j < seq && threadIdx.y == 0)
            {
                ElementAccumulator max_val = smem_max[j];
                ElementAccumulator sumexp = smem_sumexp[j];
                for(int i = 0; i < SEQ_BLOCK_SIZE; ++i)
                {
                    if(block_row + i < seq && smem_mask[i][j])
                    {
                        sumexp += ::exp(smem_QK[i][j] - max_val);
                    }
                }
                smem_sumexp[j] = sumexp;
            }
        }
        __syncthreads();
    }

    // Store result to maxsumexp
    for(int j = threadIdx.x; j < SEQ_BLOCK_SIZE; j += blockDim.x)
    {
        if(block_col + j < seq && threadIdx.y == 0)
        {
            maxsumexp[2 * (block_col + j + seq * batch_idx)] = T{smem_max[j]};
            maxsumexp[2 * (block_col + j + seq * batch_idx) + 1] = T{smem_sumexp[j]};
        }
    }
    __syncthreads();
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, T *maxsumexp) noexcept
{
    // Calculate grid size for tiles
    dim3 blocks((seq + 31)/32, batch);
    dim3 threads(32, 32);  // 1024 threads per block

    // Calculate scale factor 1/sqrt(head)
    using Y = typename T::repr_t;
    T scale = T{Y(1) / std::sqrt(Y(head))};

    // Launch kernel
    flash_maxsumexp_kernel<T><<<blocks, threads, 0, stream>>>(
        batch, seq, head, scale, K, Q, mask, maxsumexp);
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
