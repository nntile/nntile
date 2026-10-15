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
#include "nntile/kernel/cuda.hh"
#include "nntile/kernel/scale/cuda.hh"

namespace nntile::kernel::transpose
{

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
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
        __shared__ T block[BLOCK_ROW][BLOCK_COL+1];
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
            scale::cuda<T>(stream, m*n, alpha, src, dst);
        }
        else if(m < 4)
        {
            dim3 blocks(((m+1)/2) * ((n+255)/256));
            (cuda_kernel<T, 2, 256, 2>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 8)
        {
            dim3 blocks(((m+3)/4) * ((n+255)/256));
            (cuda_kernel<T, 4, 256, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 16)
        {
            dim3 blocks(((m+7)/8) * ((n+127)/128));
            (cuda_kernel<T, 8, 128, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 32)
        {
            dim3 blocks(((m+15)/16) * ((n+63)/64));
            (cuda_kernel<T, 16, 64, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
    }
    else
    {
        if(n == 1)
        {
            scale::cuda<T>(stream, m*n, alpha, src, dst);
        }
        else if(n < 4)
        {
            dim3 blocks(((m+255)/256) * ((n+1)/2));
            (cuda_kernel<T, 256, 2, 2>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 8)
        {
            dim3 blocks(((m+255)/256) * ((n+3)/4));
            (cuda_kernel<T, 256, 4, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 16)
        {
            dim3 blocks(((m+127)/128) * ((n+7)/8));
            (cuda_kernel<T, 128, 8, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 32)
        {
            dim3 blocks(((m+63)/64) * ((n+15)/16));
            (cuda_kernel<T, 64, 16, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
    }
}

//! Transpose an nr-by-nc sub-block of an m-by-n buffer on CUDA
/*! Performs the following operation for di in [0, nr) and dj in [0, nc):
 *      dst[di*n+dj] = alpha * src[di+dj*m]
 *
 * src and dst are the sub-block's base pointers (its (0, 0) corner in
 * the whole m-by-n matrix); the sub-block KEEPS the whole matrix's
 * strides - the src columns stay m elements apart, the dst rows n
 * elements apart. The chunked NNHaul wrapper drives this entry with
 * sub-blocks whose covered src column segments and whose covered dst
 * flat range each lie inside single device chunks; the whole-buffer
 * cuda() above is the (m, n) case. Internally shared-memory tiled
 * like the whole-buffer kernel: a 32x(32+1) tile with the load phase
 * coalesced along the contiguous src axis (the di axis) and the store
 * phase coalesced along the contiguous dst axis (the dj axis).
 * Elementwise and deterministic: any piece factorization is bitwise
 * the whole-buffer call.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of the whole src buffer.
 * @param[in] n: Size of the last mode of the whole src buffer.
 * @param[in] nr: Number of rows of the sub-block.
 * @param[in] nc: Number of columns of the sub-block.
 * @param[in] alpha: Scalar factor for src.
 * @param[in] src: Base pointer of the sub-block (m column stride).
 * @param[inout] dst: Base pointer of the sub-block (n row stride).
 * */
template<typename T, int TILE>
static __global__
void panel_cuda_kernel(Index m, Index n, Index nr, Index nc, Scalar alpha_,
    const T *src, T *dst)
{
    __shared__ T tile[TILE][TILE+1];
    Index const bx = blockIdx.x;
    Index const by = blockIdx.y;
    Index const tx = threadIdx.x;
    Index const ty = threadIdx.y;
    Index const rel_i = by*TILE + tx;
    Index const rel_j = bx*TILE + ty;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    // Load phase: consecutive threads read consecutive src elements
    // (the di axis is contiguous in src).
    if(rel_i < nr && rel_j < nc)
    {
        tile[tx][ty] = T{alpha * Y{src[rel_i + rel_j*m]}};
    }
    __syncthreads();
    // Store phase: consecutive threads write consecutive dst elements
    // (the dj axis is contiguous in dst); the value comes from the
    // transposed tile cell.
    if(by*TILE + ty < nr && bx*TILE + tx < nc)
    {
        dst[(by*TILE+ty)*n + bx*TILE+tx] = tile[ty][tx];
    }
}

template<typename T>
void panel(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const T* src, T* dst)
    noexcept
{
    if(nr <= 0 || nc <= 0)
    {
        return;
    }
    constexpr int TILE = 32;
    dim3 threads(TILE, TILE);
    dim3 blocks((nc+TILE-1)/TILE, (nr+TILE-1)/TILE);
    (panel_cuda_kernel<T, TILE>)<<<blocks, threads, 0, stream>>>(m, n, nr,
            nc, alpha, src, dst);
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

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp16_t* src, fp16_t* dst)
    noexcept;

template
void panel<fp32_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp32_t* src, fp32_t* dst)
    noexcept;

template
void panel<fp64_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp64_t* src, fp64_t* dst)
    noexcept;

template
void panel<bf16_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const bf16_t* src, bf16_t* dst)
    noexcept;

template
void panel<fp16_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp16_t* src, fp16_t* dst)
    noexcept;

} // namespace nntile::kernel::tranpose
