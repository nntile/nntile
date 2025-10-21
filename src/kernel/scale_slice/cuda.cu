/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scale_slice/cuda.cu
 * Per-element scaling of a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scale_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::scale_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, T *dst)
//! Per-element scaling of a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] mk: Product of m and k
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to set in the output fiber
        const Y src_val = Y{alpha} * Y{src[i1*m+i0]};
        // Set output value
        dst_fiber[i2*m] = T{src_val};
    }
}

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
static __global__
void cuda_kernel_optimized(Index m, Index n, Index k, Scalar alpha_, const T *src,
        T *dst)
//! Per-element scaling of a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    Index griddim_m = (m+BLOCK_M-1) / BLOCK_M;
    Index griddim_k = (k+BLOCK_K-1) / BLOCK_K;
    Index block_i = blockIdx.x % griddim_m;
    Index block_lj = blockIdx.x / griddim_m;
    Index block_l = block_lj % griddim_k;
    Index block_j = block_lj / griddim_k;
    Index i = block_i*BLOCK_M + threadIdx.x;
    Index l = block_l * BLOCK_K;
    Index j = block_j * BLOCK_N;
    for(Index ii = i; ii < ::min(m, i+BLOCK_M); ii += blockDim.x)
    {
        for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
        {
            T src_val = static_cast<T>(alpha * static_cast<Y>(src[ii+jj*m]));
            for(Index ll = l; ll < ::min(k, l+BLOCK_K); ++ll)
            {
                dst[ii+m*(ll+k*jj)] = src_val;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src_, T *dst_)
    noexcept
//! Per-element scaling of a broadcasted slice on CUDA
/*! This is a host function that does the following operations:
 *      dst[i,l,j] = alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for src
 * @param[in] src_: Input contiguous m-by-n array
 * @param[out] dst_: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Optimized version for large m and small n and k
    if(m > 1024 or n > 4 or k > 4)
    {
        dim3 threads(256);
        if(n == 1)
        {
            dim3 blocks(((m+1023)/1024) * ((k+3)/4));
            (cuda_kernel_optimized<T, 1024, 1, 4>)<<<blocks, threads, 0, stream>>>(m,
                    n, k, alpha, src_, dst_);
        }
        else if(n == 2)
        {
            dim3 blocks(((m+1023)/1024) * ((k+3)/4));
            (cuda_kernel_optimized<T, 1024, 2, 4>)<<<blocks, threads, 0, stream>>>(m,
                    n, k, alpha, src_, dst_);
        }
        else if(n == 3)
        {
            dim3 blocks(((m+1023)/1024) * ((k+3)/4));
            (cuda_kernel_optimized<T, 1024, 3, 4>)<<<blocks, threads, 0, stream>>>(m,
                    n, k, alpha, src_, dst_);
        }
        else
        {
            dim3 blocks(((m+1023)/1024) * ((k+3)/4) * ((n+3)/4));
            (cuda_kernel_optimized<T, 1024, 4, 4>)<<<blocks, threads, 0, stream>>>(m,
                    n, k, alpha, src_, dst_);
        }
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src_, dst_);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_fast_tf32_t *src, fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_fast_fp16_t *src, fp32_fast_fp16_t *dst)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_fast_bf16_t *src, fp32_fast_bf16_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::scale_slice
