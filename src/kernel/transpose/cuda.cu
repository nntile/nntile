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
 * @version 1.0.0
 * */

#include "nntile/kernel/transpose/cuda.hh"
#include <algorithm>
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
    Index i = threadIdx.x + blockIdx.x*blockDim.x;
    Index j = i / m;
    i = i - j*m;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};

    if(i < m and j < n)
    {
        dst[i*n+j] = T{alpha * Y{src[i+j*m]}};
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
    dim3 threads(32);
    dim3 blocks((m*n+threads.x-1)/threads.x);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, alpha, src, dst);
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
