/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_softmax_gemm_backward_sumprod_slice/cuda.cu
 * CUDA kernel to compute backward pass of softmax(A)V with fused sumprod and slice
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/flash_softmax_gemm_backward_sumprod_slice/cuda.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace nntile::kernel::flash_softmax_gemm_backward_sumprod_slice
{

template<typename T>
__global__ void flash_softmax_gemm_backward_sumprod_slice_kernel(
        Index batch, Index seq, Index head,
        T scale, const T *K, const T *Q, const bool_t *mask,
        const T *maxsumexp, const T *dA, const T *V, T *dV, T *sumprod_slice)
{
    using Y = typename T::repr_t;
    // Get global thread indices
    const Index batch_idx = blockIdx.y;
    const Index seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Index tid = threadIdx.x;

    constexpr Y inf = Y{1.0} / Y{0.0};

    // Shared memory for K, V and softmax values
    __shared__ Y K_shared[256];  // One row of K at a time
    __shared__ Y V_shared[256];  // One row of V at a time
    __shared__ Y softmax_shared[256];  // Softmax values for the block

    if (seq_idx < seq)
    {
        // Calculate base offsets in Fortran order [head, seq, batch]
        const Index Q_base = head * (seq_idx + seq * batch_idx);
        const Index maxsumexp_base = 2*(seq_idx + seq * batch_idx);

        // Get pre-computed max and sumexp values
        const Y max_val = Y{maxsumexp[maxsumexp_base]};
        const Y sumexp = Y{maxsumexp[maxsumexp_base + 1]};

        // Initialize sumprod accumulator
        Y sumprod = 0;

        // First compute softmax(QK^T/sqrt(d))
        for (Index j = 0; j < seq; ++j)
        {
            // Load K values for this j into shared memory
            const Index K_base = head * (j + seq * batch_idx);
            for (Index d = tid; d < head; d += blockDim.x)
            {
                K_shared[d] = Y{K[K_base + d]};
            }
            __syncthreads();

            // Compute dot product Q[h,i,b] * K[h,j,b]
            Y dot_prod = 0;
            for (Index d = 0; d < head; ++d)
            {
                dot_prod += Y{Q[d + Q_base]} * K_shared[d];
            }
            dot_prod *= Y{scale};

            // Apply mask and compute softmax value
            Y softmax_val;
            if (mask != nullptr && !mask[j + seq * seq_idx])
            {
                softmax_val = 0;
            }
            else
            {
                softmax_val = ::exp(dot_prod - max_val) / sumexp;
            }
            softmax_shared[tid] = softmax_val;
            __syncthreads();

            // Load dA[h,i,b] into shared memory
            const Index dA_base = head * (j + seq * batch_idx);  // [head, seq, batch]
            if (tid < head) {
                K_shared[tid] = Y{dA[dA_base + tid]};  // Load dA[h,i,b]
            }
            __syncthreads();

            // Update dV = dA * softmax^T
            // For position (i,j):
            // - dA is [head, seq, batch] in Fortran order
            // - softmax is [seq, seq] in Fortran order, but we need its transpose
            // - dV is [head, seq, batch] in Fortran order
            // So we accumulate dA[h,i,b] * softmax[j,i] into dV[h,i,b]
            for (Index d = 0; d < head; ++d) {
                const Index dV_idx = d + head * (j + seq * batch_idx);  // [head, seq, batch]
                Y dv_val = K_shared[d] * softmax_val;  // dA[h,i,b] * softmax[j,i]
                if constexpr (std::is_same_v<Y, float>) {
                    atomicAdd(reinterpret_cast<float*>(&dV[dV_idx]), dv_val);
                } else if constexpr (std::is_same_v<Y, double>) {
                    atomicAdd(reinterpret_cast<double*>(&dV[dV_idx]), dv_val);
                }
            }

            // Accumulate tmp_grad * softmax into sumprod
            // tmp_grad = V^T * dA
            Y tmp_grad = 0;
            for (Index d = 0; d < head; ++d) {
                tmp_grad += Y{V[d + head * (j + seq * batch_idx)]} * K_shared[d];  // V[h,j,b] * dA[h,i,b]
            }
            sumprod += tmp_grad * softmax_val;
        }

        // Store sumprod result
        // if (tid == 0) {
        //     sumprod_slice[seq_idx + seq * batch_idx] = T{sumprod};
        // }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *dA, const T *V, T *dV, T *sumprod_slice) noexcept
{
    // Calculate scaling factor
    using Y = typename T::repr_t;
    Y scale_val = Y(1.0) / std::sqrt(Y(head));
    T scale = T{scale_val};

    // Configure kernel launch parameters
    const int block_size = 256;
    const int blocks_per_seq = (seq + block_size - 1) / block_size;

    dim3 grid(blocks_per_seq, batch);
    dim3 block(block_size);

    // Launch kernel
    flash_softmax_gemm_backward_sumprod_slice_kernel<T><<<grid, block, 0, stream>>>(
        batch, seq, head, scale, K, Q, mask, maxsumexp, dA, V, dV, sumprod_slice);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_t *K, const fp32_t *Q, const bool_t *mask,
        const fp32_t *maxsumexp, const fp32_t *dA, const fp32_t *V,
        fp32_t *dV, fp32_t *sumprod_slice) noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp64_t *K, const fp64_t *Q, const bool_t *mask,
        const fp64_t *maxsumexp, const fp64_t *dA, const fp64_t *V,
        fp64_t *dV, fp64_t *sumprod_slice) noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const bf16_t *K, const bf16_t *Q, const bool_t *mask,
        const bf16_t *maxsumexp, const bf16_t *dA, const bf16_t *V,
        bf16_t *dV, bf16_t *sumprod_slice) noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_tf32_t *K, const fp32_fast_tf32_t *Q, const bool_t *mask,
        const fp32_fast_tf32_t *maxsumexp, const fp32_fast_tf32_t *dA,
        const fp32_fast_tf32_t *V, fp32_fast_tf32_t *dV,
        fp32_fast_tf32_t *sumprod_slice) noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_fp16_t *K, const fp32_fast_fp16_t *Q, const bool_t *mask,
        const fp32_fast_fp16_t *maxsumexp, const fp32_fast_fp16_t *dA,
        const fp32_fast_fp16_t *V, fp32_fast_fp16_t *dV,
        fp32_fast_fp16_t *sumprod_slice) noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index batch, Index seq, Index head,
        const fp32_fast_bf16_t *K, const fp32_fast_bf16_t *Q, const bool_t *mask,
        const fp32_fast_bf16_t *maxsumexp, const fp32_fast_bf16_t *dA,
        const fp32_fast_bf16_t *V, fp32_fast_bf16_t *dV,
        fp32_fast_bf16_t *sumprod_slice) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm_backward_sumprod_slice
