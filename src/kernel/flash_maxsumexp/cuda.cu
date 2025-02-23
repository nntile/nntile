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

namespace nntile::kernel::flash_maxsumexp
{

template<typename T>
__global__ void flash_maxsumexp_kernel(Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask, T *maxsumexp)
{
    using Y = typename T::repr_t;
    // Get global thread indices
    const Index batch_idx = blockIdx.y;
    const Index seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Index tid = threadIdx.x;

    constexpr Y inf = Y{1.0} / Y{0.0};

    // Shared memory for K values used by the block
    __shared__ Y K_shared[256];  // One row of K at a time

    if (seq_idx < seq)
    {
        // Calculate base offsets for Q and K
        // In Fortran order: [head, seq, batch]
        // Q[h,i,b] = Q[h + head*(i + seq*b)]
        const Index Q_base = head * (seq_idx + seq * batch_idx);

        // Initialize max value and sum
        Y max_val = -inf;
        Y sum = 0;
        Y running_max = -inf;

        // Single pass: compute dot products, update max and running sum
        for (Index j = 0; j < seq; ++j)
        {
            // Load K values for this j into shared memory
            const Index K_base = head * (j + seq * batch_idx);
            // Each thread loads multiple elements to cover up to 256 head size
            for (Index d = tid; d < head; d += blockDim.x)
            {
                K_shared[d] = Y{K[K_base + d]};
            }
            __syncthreads();

            // Compute dot product Q[h,i,b] * K[h,j,b]
            Y dot_prod = 0;

            // Dot product using shared memory
            for (Index d = 0; d < head; ++d)
            {
                dot_prod += Y{Q[d + Q_base]} * K_shared[d];
            }
            dot_prod *= Y{scale};

            // Apply mask if present (mask is [seq, seq] in Fortran order)
            if (mask != nullptr && !mask[j + seq * seq_idx])
            {
                dot_prod = -inf;
            }

            // Update max and sum
            Y old_max = running_max;
            running_max = ::max(running_max, dot_prod);

            // Compute sum of exponentials depending on the max value
            if (running_max == old_max)
            {
                // If max didn't change, just add to sum
                sum += ::exp(dot_prod - running_max);
            }
            else
            {
                // If we found a new max, rescale old sum and add 1
                sum = sum * ::exp(old_max - running_max) + Y{1.0};
            }

            __syncthreads();  // Ensure shared memory is ready for next iteration
        }

        // Calculate output index for [2, seq, batch] Fortran order
        // maxsumexp[t,i,b] = maxsumexp[t + 2*(i + seq*b)]
        const Index out_base = 2*(seq_idx + seq * batch_idx);
        maxsumexp[out_base] = T{running_max};     // First element (max)
        maxsumexp[out_base+1] = T{sum};          // Second element (sumexp)
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, T *maxsumexp) noexcept
{
    // Calculate scaling factor
    using Y = typename T::repr_t;
    T scale = T(Y(1.0) / std::sqrt(Y(head)));

    // Configure kernel launch parameters
    const int block_size = 256;
    const int blocks_per_seq = (seq + block_size - 1) / block_size;

    dim3 grid(blocks_per_seq, batch);
    dim3 block(block_size);

    // Launch kernel with Q and K swapped to match kernel signature
    flash_maxsumexp_kernel<T><<<grid, block, 0, stream>>>(batch, seq, head, scale,
        K, Q, mask, maxsumexp);
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
