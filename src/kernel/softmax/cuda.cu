/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
 * @date 2022-12-08
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
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    // Outer loop by the last mode of dst and sumnorm arrays
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Middle loop by the middle mode of dst array
        for(Index i1 = i1_start; i1 < k; i1 += i1_step)
        {
            Index src_offset = 2 * m * i2;
            Index dst_offset = (i2*k+i1) * m;
            // Inner loop by the first mode of dst and sumnorm arrays
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T &val = dst[dst_offset];
                // Max and sum of exponents
                const T max = maxsumexp[src_offset];
                const T sum = maxsumexp[src_offset+1];
                // Update value
                val = exp(val-max) / sum;
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
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
    dim3 blocks(16, 16), threads(8, 4);
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

