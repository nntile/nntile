/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/softmax_inplace/cuda.cu
 * Inplace softmax operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/softmax_inplace/cuda.hh"
#include <stdio.h>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::softmax_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index m_per_block, Index n, Index n_per_block,
        Index k, const T *maxsumexp, Scalar alpha, T *dst)
{
    Index i0_block = blockIdx.y, i1_block = blockIdx.z,
          i2_start = threadIdx.x, i2_step = blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};

    for(Index i0 = i0_block*m_per_block;
            i0 < (i0_block+1)*m_per_block and i0 < m; ++i0)
    {
        for(Index i1 = i1_block*n_per_block;
                i1 < (i1_block+1)*n_per_block and i1 < n; ++i1)
        {
            Index dst_offset = i1*k*m + i0;
            T *dst_slice = dst + dst_offset;
            // Max and sum of exponents
            __shared__ Y max, sum;
            if(i2_start == 0)
            {
                Index src_offset = m*i1 + i0;
                max = Y{maxsumexp[2*src_offset]};
                sum = Y{maxsumexp[2*src_offset+1]};
            }
            __syncthreads();
            for(Index i2 = i2_start; i2 < k; i2 += i2_step)
            {
                // Value-to-update
                T &val = dst_slice[i2*m];
                // Update value
                if(not ::isinf(Y{val}))
                {
                    val = T{alpha * ::exp(Y{val}-max) / sum};
                }
                else
                {
                    val = T{zero};
                }
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        Scalar alpha, T *dst)
    noexcept
//! softmax of a buffer along middle axis
/*!
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] maxsumexp: Maximums and sums of exponents of slices
 * @param[in] alpha: Scalar multiplier for the output
 * @param[in] dst: Contiguous output array
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 threads(32, 1, 1), blocks(1, m, n);
    Index m_per_block = 1, n_per_block = 1;
    if(m > 65535)
    {
        m_per_block = (m+65534) / 65535;
        blocks.y = (m+m_per_block-1) / m_per_block;
    }
    if(n > 65535)
    {
        n_per_block = (n+65534) / 65535;
        blocks.z = (n+n_per_block-1) / n_per_block;
    }
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, m_per_block, n,
            n_per_block, k, maxsumexp, alpha, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *maxsumexp, Scalar alpha, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *maxsumexp, Scalar alpha, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *maxsumexp, Scalar alpha, bf16_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_fast_tf32_t *maxsumexp, Scalar alpha, fp32_fast_tf32_t *dst)
    noexcept;

} // namespace nntile::kernel::softmax_inplace
