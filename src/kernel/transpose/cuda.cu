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

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel2(Index m, Index n, Scalar alpha_, const T *src, T *dst)
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
    Index src_i = threadIdx.x % BLOCK_ROW;
    Index src_j = threadIdx.x / BLOCK_ROW;
    Index src_griddim_row = (m+BLOCK_ROW-1) / BLOCK_ROW;
    Index src_block_i = blockIdx.x % src_griddim_row;
    Index src_block_j = blockIdx.x / src_griddim_row;
    Index global_src_i = src_i + src_block_i*BLOCK_ROW;
    Index global_src_j = src_j + src_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
 
    if((src_block_i+1)*BLOCK_ROW <= m and (src_block_j+1)*BLOCK_COL <= n)
    {
        __shared__ T block[BLOCK_ROW][BLOCK_COL];
        const T *src_slice = src + global_src_i + global_src_j*m;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        for(int k = 0; k < BLOCK_COL; k += BLOCK_COL_STEP)
        {
            block[src_i][src_j+k] = T{alpha * Y{src_slice[k*m]}};
        }
        Index dst_i = threadIdx.x % BLOCK_COL;
        Index dst_j = threadIdx.x / BLOCK_COL;
        Index dst_block_i = src_block_j;
        Index dst_block_j = src_block_i;
        Index global_dst_i = dst_i + dst_block_i*BLOCK_COL;
        Index global_dst_j = dst_j + dst_block_j*BLOCK_ROW;
        T *dst_slice = dst + global_dst_i + global_dst_j*n;
        __syncthreads();
        constexpr int BLOCK_ROW_STEP = BLOCK_ROW / BLOCK_LOOP;
        for(int k = 0; k < BLOCK_ROW; k += BLOCK_ROW_STEP)
        {
            dst_slice[k*n] = block[dst_j+k][dst_i];
        }
    }
    else if(global_src_i < m)
    {
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        for(Index new_j = 0; new_j < BLOCK_COL; new_j += BLOCK_COL_STEP)
        {
            if(global_src_j+new_j >= n)
            {
                break;
            }
            dst[global_src_j+new_j+global_src_i*n] =
                 T{alpha * Y{src[global_src_i+(global_src_j+new_j)*m]}};
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
    dim3 threads(256);
    if(m < n)
    {
        if(m == 1)
        {
            cudaMemcpyAsync(dst, src, sizeof(*src)*n, cudaMemcpyDeviceToDevice,
                    stream);
        }
        else if(m < 4)
        {
            dim3 blocks(((m+1)/2) * ((n+511)/512));
            (cuda_kernel2<T, 2, 512, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 8)
        {
            dim3 blocks(((m+3)/4) * ((n+255)/256));
            (cuda_kernel2<T, 4, 256, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 16)
        {
            dim3 blocks(((m+7)/8) * ((n+127)/128));
            (cuda_kernel2<T, 8, 128, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 32)
        {
            dim3 blocks(((m+15)/16) * ((n+63)/64));
            (cuda_kernel2<T, 16, 64, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel2<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
    }
    else
    {
        if(n == 1)
        {
            cudaMemcpyAsync(dst, src, sizeof(*src)*m, cudaMemcpyDeviceToDevice,
                    stream);
        }
        else if(n < 4)
        {
            dim3 blocks(((m+511)/512) * ((n+1)/2));
            (cuda_kernel2<T, 512, 2, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 8)
        {
            dim3 blocks(((m+255)/256) * ((n+3)/4));
            (cuda_kernel2<T, 256, 4, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 16)
        {
            dim3 blocks(((m+127)/128) * ((n+7)/8));
            (cuda_kernel2<T, 128, 8, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 32)
        {
            dim3 blocks(((m+63)/64) * ((n+15)/16));
            (cuda_kernel2<T, 64, 16, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel2<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
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
