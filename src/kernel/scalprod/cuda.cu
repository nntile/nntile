/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scalprod/cuda.cc
 * Scalar product of buffers on GPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#include "nntile/kernel/scalprod/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace scalprod
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src1,
        const T *src2, T beta, T *dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    // Cycle over row of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over column of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Get scalar product of corresponding slices
            const T *src1_slice = src1 + i2*mk + i1;
            const T *src2_slice = src2 + i2*mk + i1;
            // Offset for output
            Index dst_offset = i1 + i2*m;
            // Init scalar product
            T sum = 0.0;
            // Cycle over slices of inputs
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from sources
                T val1 = src1_slice[i0*m];
                T val2 = src2_slice[i0*m];
                // Accumulate scalar product
                sum += val1 * val2;
            }
            // Save result
            if(beta == 0.0)
            {
                dst[dst_offset] = alpha * sum;
            }
            else
            {
                dst[dst_offset] = beta*dst[dst_offset] + alpha*sum;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src1, const T *src2, T beta, T *dst)
    noexcept
//! Scalar product along middle axis
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute scalar
 * products of slices along second axis with k elements, resulting in m-by-n
 * output array dst. If beta is non-zero, then values of array dst are updated
 * by this routine in read-write mode, therefore dst must be initialized before
 * use. If beta is zero, all values of dst are overwritten and it shall be used
 * in write mode.
 *
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*src1[i,:,j]@src2[i,:,j]
 *      
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      scalar products of src1 and src2 along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha, src1,
            src2, beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src1, const fp32_t *src2, fp32_t beta, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src1, const fp64_t *src2, fp64_t beta, fp64_t *sum_dst)
    noexcept;

} // namespace scalprod
} // namespace kernel
} // namespace nntile

