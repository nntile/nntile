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
 * @date 2023-05-03
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
void cuda_kernel(Index m, Index n, Index k, Index mk, const T *maxsumexp,
        T *dst)
{
    int i1 = threadIdx.x + blockIdx.x*blockDim.x,
        i2 = threadIdx.y + blockIdx.y*blockDim.y;
    // Check column index of output buffer
    if(i2 < n)
    {
        // Check row index of output buffer
        if(i1 < m)
        {
            // Output fiber
            T *dst_fiber = dst + i2*mk + i1;
            // Max and sum of exponents
            Index src_offset = 2 * (i1+i2*m);
            T max = maxsumexp[src_offset];
            T sum = maxsumexp[src_offset+1];
            // Cycle over output fiber
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Value-to-update
                T &val = dst_fiber[i0*m];
                // Update value
                val = exp(val-max) / sum;
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
    // Both source and destination are Fortran-contiguous
    dim3 blocks((m+15)/16, (n+15)/16), threads(16, 16);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, maxsumexp,
            dst);
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

