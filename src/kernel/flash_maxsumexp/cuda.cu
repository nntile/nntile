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

template<typename T, typename Y, Index SEQ_BLOCK_SIZE, Index HEAD_SIZE>
static __device__ void gemm_smem_sync(const T smem_K[SEQ_BLOCK_SIZE][HEAD_SIZE],
        const T smem_Q[HEAD_SIZE][SEQ_BLOCK_SIZE],
        const bool_t smem_mask[SEQ_BLOCK_SIZE][SEQ_BLOCK_SIZE],
        T scale,
        Y smem_QK[SEQ_BLOCK_SIZE][SEQ_BLOCK_SIZE])
{
    using namespace nvcuda::wmma;
    // Special case for nntile::fp32_fast_tf32_t using WMMA
    if constexpr (std::is_same<T, nntile::fp32_fast_tf32_t>::value)
    {
        auto smem_K_float = reinterpret_cast<const float *>(smem_K);
        auto smem_Q_float = reinterpret_cast<const float *>(smem_Q);
        // Define fragments using float precision TF32 supports only 16x16x8 grid
        fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
        fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
        fragment<accumulator, 16, 16, 8, float> c_frag;

        // Load data into fragments and perform matrix multiplication
        for(int i = 0; i < SEQ_BLOCK_SIZE; i += 16)
        {
            for(int j = 0; j < SEQ_BLOCK_SIZE; j += 16)
            {
                // Initialize accumulator fragment
                fill_fragment(c_frag, 0.0f);
                for(int k = 0; k < HEAD_SIZE; k += 8)
                {
                    // Load matrix A and B from shared memory into fragments
                    // K[SEQ_BLOCK_SIZE, HEAD_SIZE] - leading dimension is HEAD_SIZE
                    load_matrix_sync(a_frag,
                        smem_K_float + i * HEAD_SIZE + k, HEAD_SIZE);
                    // Q[HEAD_SIZE, SEQ_BLOCK_SIZE] - leading dimension is SEQ_BLOCK_SIZE
                    load_matrix_sync(b_frag,
                        smem_Q_float + k * SEQ_BLOCK_SIZE + j, SEQ_BLOCK_SIZE);
                    // Perform matrix multiplication and accumulate
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                // Apply scale factor to the result
                for(int k = 0; k < c_frag.num_elements; ++k)
                {
                    c_frag.x[k] *= float{scale};
                }
                __syncthreads();
                // Store results back to shared memory
                store_matrix_sync(&smem_QK[i][j], c_frag, SEQ_BLOCK_SIZE, mem_row_major);
            }
        }
    }
    // General case
    else
    {
        for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
            {
                Y sum = 0;
                for(int k = 0; k < HEAD_SIZE; ++k)
                {
                    // K[i,k] * Q[k,j] for K @ Q
                    sum += Y(smem_K[i][k]) * Y(smem_Q[k][j]);
                }
                smem_QK[i][j] = sum * Y(scale);
            }
        }
        __syncthreads();
    }
}

template<typename T, Index SEQ_BLOCK_SIZE, Index HEAD_SIZE>
__global__ void flash_maxsumexp_kernel(Index batch, Index seq,
        T scale, const T *K, const T *Q, const bool_t *mask, T *maxsumexp)
{
    using Scalar = typename T::repr_t;
    // Accumulator is fp64_t for fp64_t input type and fp32_t for other input types
    using ElementAccumulator = typename std::conditional<
        std::is_same<T, nntile::fp64_t>::value, double, float>::type;

    // Shared memory
    __shared__ T smem_Q[HEAD_SIZE][SEQ_BLOCK_SIZE];
    __shared__ T smem_K[SEQ_BLOCK_SIZE][HEAD_SIZE];
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

    // Load Q[0:HEAD_SIZE, block_row:block_row+SEQ_BLOCK_SIZE, batch_idx]
    for(int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x)
    {
        for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
        {
            if(block_col + j < seq)
            {
                smem_Q[i][j] = Q[i + HEAD_SIZE * (block_col + j) + HEAD_SIZE * seq * batch_idx];
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

        // Load K[0:HEAD_SIZE, block_col:block_col+SEQ_BLOCK_SIZE, batch_idx]
        for(int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x)
        {
            for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
            {
                if(block_row + j < seq)
                {
                    smem_K[j][i] = K[i + HEAD_SIZE * (block_row + j) + HEAD_SIZE * seq * batch_idx];
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
        gemm_smem_sync<T, ElementAccumulator, SEQ_BLOCK_SIZE, HEAD_SIZE>(smem_K,
                smem_Q, smem_mask, scale, smem_QK);
        // __syncthreads();
        // for(int i = threadIdx.x; i < SEQ_BLOCK_SIZE; i += blockDim.x)
        // {
        //     for(int j = threadIdx.y; j < SEQ_BLOCK_SIZE; j += blockDim.y)
        //     {
        //         smem_QK[i][j] *= ElementAccumulator(scale);
        //     }
        // }

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
    dim3 blocks((seq + 15)/16, batch);
    dim3 threads(16, 16);  // 256 threads per block

    // Calculate scale factor 1/sqrt(head)
    using Y = typename T::repr_t;
    T scale = T{Y(1) / std::sqrt(Y(head))};

    // Launch kernel
    if(head == 64)
    {
        flash_maxsumexp_kernel<T, 16, 64><<<blocks, threads, 0, stream>>>(
            batch, seq, scale, K, Q, mask, maxsumexp);
    }
    else if(head == 128)
    {
        flash_maxsumexp_kernel<T, 16, 128><<<blocks, threads, 0, stream>>>(
            batch, seq, scale, K, Q, mask, maxsumexp);
    }
    else if(head == 256)
    {
        flash_maxsumexp_kernel<T, 16, 256><<<blocks, threads, 0, stream>>>(
            batch, seq, scale, K, Q, mask, maxsumexp);
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
