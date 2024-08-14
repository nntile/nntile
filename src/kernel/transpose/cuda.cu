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
    Index block_i = blockIdx.x;
    Index block_j = blockIdx.y;
    Index global_i = i + block_i*blockDim.x;
    Index global_j = j + block_j*blockDim.y;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};

    if(global_i < m and global_j < n)
    {
        __shared__ T block[64];
        //dst[i*n+j] = T{alpha * Y{src[i+j*m]}};
        block[i+j*8] = T{alpha * Y{src[global_i + global_j*m]}};
        __syncthreads();
        dst[global_i*n + global_j] = block[i*8+j];
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
    dim3 threads(8, 8);
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, alpha, src, dst);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess)
    {
        std::cerr << "Error in src::kernel::transpose::cuda<T>\n";
        std::cerr << "m=" << m << " n=" << n << "\n";
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
