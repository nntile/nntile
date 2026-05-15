/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_infnan/cuda.cu
 * Accumulate flags for NaN and Inf in a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_infnan/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::accumulate_infnan
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, Index *dst)
{
    using Y = typename T::repr_t;
    // Index current_flag = 0;

    // // Shared memory for block reduction
    // __shared__ Index shared_flag[1024];

    // int tid = threadIdx.x;

    // Y current_val = Y{0};

    // // Each thread computes partial sum using Kahan summation
    // for(Index i = tid; i < nelems; i += blockDim.x)
    // {
    //     current_val = Y{src[i]};
    //     if(!isfinite(current_val))
    //     {
    //         current_flag = 1;
    //         break;
    //     }
    // }
    // // Store partial results in shared memory
    // shared_flag[tid] = current_flag;
    // __syncthreads();

    // // Reduce within block using tree reduction with Kahan summation
    // for(int s = blockDim.x/2; s > 0; s >>= 1)
    // {
    //     if(tid < s)
    //     {
    //         // Accumulate shared_c[tid+s]
    //         Y y = shared_c[tid + s] - shared_c[tid];
    //         Y t = shared_sum[tid] + y;
    //         shared_c[tid] = (t - shared_sum[tid]) - y;
    //         shared_sum[tid] = t;
    //         // Accumulate shared_sum[tid+s]
    //         y = shared_sum[tid + s] - shared_c[tid];
    //         t = shared_sum[tid] + y;
    //         shared_c[tid] = (t - shared_sum[tid]) - y;
    //         shared_sum[tid] = t;
    //     }
    //     __syncthreads();
    // }

    // // Final result computation by thread 0
    // if(tid == 0)
    // {
    //     const Y sum_val = alpha * shared_sum[0];
    //     if(beta == zero)
    //     {
    //         dst[0] = static_cast<T>(sum_val);
    //     }
    //     else
    //     {
    //         dst[0] = static_cast<T>(
    //             (beta * Y{dst[0]} - alpha*shared_c[0]) + sum_val);
    //     }
    // }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Index current_flag = 0;
    if(idx < nelems)
    {
        Y val = Y{src[idx]};
        // Check for NaN or Inf
        if(isnan(val) || isinf(val))
        {
            current_flag = 1;
        }
    }
    if((current_flag == 1) && (dst[0] == 0))
    {
        dst[0] = current_flag;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, Index *dst)
    noexcept
//! Accumulate flags for Inf and NaN elements in buffer
/*! For a provided input array of nelems elements indicate 
 *  whether there is NaN of Inf elements

 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Number of elements in the input array
 * @param[in] src: Input contiguous array
 * @param[inout] dst: Output scalar (single element array)
 * */
{
    // Use a single block with up to 1024 threads
    dim3 threads(1024);
    dim3 blocks(1);

    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t *src, Index *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t *src, Index *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t *src, Index *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t *src, Index *dst)
    noexcept;

} // namespace nntile::kernel::accumulate_infnan
