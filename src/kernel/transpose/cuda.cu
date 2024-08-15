/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cuda.cu
 * Transpose operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/transpose/cuda.hh"
#include <algorithm>
#include <iostream>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::transpose
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Scalar alpha_, const T *src, T *dst)
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha_: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    Index i = threadIdx.x;
    Index j = threadIdx.y;
    Index griddim_x = (m+15) / 16;
    Index block_i = blockIdx.x % griddim_x;
    Index block_j = blockIdx.x / griddim_x;
    Index global_i = i + block_i*16;
    Index global_j = j + block_j*16;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
 
    if(block_i*16+15 < m and block_j*16+15 < n)
    {
        __shared__ T block[16][17];
        const T *src_slice = src + global_i + global_j*m;
        block[i][j] = T{alpha * Y{src_slice[0]}};
        block[i][j+4] = T{alpha * Y{src_slice[4*m]}};
        block[i][j+8] = T{alpha * Y{src_slice[8*m]}};
        block[i][j+12] = T{alpha * Y{src_slice[12*m]}};
        Index dst_i = i;
        Index dst_j = j;
        Index global_dst_i = dst_i + block_j*16;
        Index global_dst_j = dst_j + block_i*16;
        T *dst_slice = dst + global_dst_i + global_dst_j*n;
        __syncthreads();
        dst[0] = block[dst_j][dst_i];
        dst[4*n] = block[dst_j+4][dst_i];
        dst[8*n] = block[dst_j+8][dst_i];
        dst[12*n] = block[dst_j+12][dst_i];
    }
    else if(global_i < m)
    {
        for(Index new_j = 0; new_j < 16; new_j += 4)
        {
            if(global_j+new_j >= n)
            {
                break;
            }
            dst[global_j+new_j+global_i*n] =
                 T{alpha * Y{src[global_i+(global_j+new_j)*m]}};
        }
    }
}

template<typename T>
static __global__
void cuda_kernel_thin(Index m, Index n, Scalar alpha_, const T *src, T *dst)
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha_: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    Index i = threadIdx.x;
    Index j = threadIdx.y;
    Index block_j = blockIdx.x;
    Index global_j = j + block_j*blockDim.y;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
 
    if(block_j*blockDim.y+blockDim.y <= n)
    {
        __shared__ T block[16][17];
        const T *src_slice = src + i + global_j*m;
        block[i][j] = T{alpha * Y{src_slice[0]}};
        Index src_offset = i + j*m;
        Index dst_i = src_offset % blockDim.y;
        Index dst_j = src_offset / blockDim.y;
        T *dst_slice = dst + block_j*blockDim.y + dst_i + dst_j*n;
        __syncthreads();
        dst_slice[0] = block[dst_j][dst_i];
    }
    else if(global_j < n)
    {
        dst[global_j+i*n] = T{alpha * Y{src[i+global_j*m]}};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Scalar alpha, const T *src,
        T *dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    // Both source and destination are Fortran-contiguous
    if(m >= 16)
    {
        dim3 threads(16, 4);
        dim3 blocks(((m+15)/16) * ((n+15)/16));
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, alpha, src, dst);
    }
    else if(m > 1)
    {
        dim3 threads(m, (31+m)/m);
        dim3 blocks((n+threads.y-1)/threads.y);
        (cuda_kernel_thin<T>)<<<blocks, threads, 0, stream>>>(m, n, alpha, src, dst);
    }
    else
    {
        cudaMemcpyAsync(dst, src, sizeof(*src)*n, cudaMemcpyDeviceToDevice, stream);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp32_t* src, fp32_t* dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp64_t* src, fp64_t* dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const bf16_t* src, bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::tranpose
