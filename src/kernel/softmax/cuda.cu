/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/softmax/cuda.cu
 * Softmax operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/softmax/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace softmax
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, const T *maxsumexp, T *dst)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2_start = threadIdx.z, i2_step = blockDim.z;
    if(i0 < m and i1 < n)
    {
        Index dst_offset = i1*k*m + i0;
        T *dst_slice = dst + dst_offset;
        // Max and sum of exponents
        __shared__ T max, sum;
        if(i2_start == 0)
        {
            Index src_offset = m*i1 + i0;
            max = maxsumexp[2*src_offset];
            sum = maxsumexp[2*src_offset+1];
        }
        __syncthreads();
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            // Value-to-update
            T &val = dst_slice[i2*m];
            // Update value
            val = ::exp(val-max) / sum;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        T *dst)
    noexcept
//! Softmax of a buffer along middle axis
/*!
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] maxsumexp: Maximums and sums of exponents of slices
 * @param[in] dst: Contiguous output array
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 threads(1, 1, 64);
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, maxsumexp, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *maxsumexp, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *maxsumexp, fp64_t *dst)
    noexcept;

} // namespace softmax
} // namespace kernel
} // namespace nntile

