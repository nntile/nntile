/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cuda.cu
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/maxsumexp/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace maxsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk,
        const T * __restrict__ src, T * __restrict__ maxsumexp)
{
    Index i1 = threadIdx.x + blockIdx.x*blockDim.x,
          i2 = threadIdx.y + blockIdx.y*blockDim.y,
          i0_start = threadIdx.z, i0_step = blockDim.z;
    constexpr T zero = 0, one = 1;
    if(i2 < n and i1 < m)
    {
        // Get max and sum of exponents of a corresponding slice
        const T *src_slice = src + i2*mk + i1;
        // Init max and sum
        T max_val = src_slice[0];
        T sum_val = zero;
        // Cycle over slice of input buffer
        for(Index i0 = i0_start; i0 < k; i0 += i0_step)
        {
            // Read value from source
            T val = src_slice[i0*m];
            // Update max and sum of exponents
            if(max_val < val)
            {
                sum_val = sum_val*(::exp(max_val-val)) + one;
                max_val = val;
            }
            else
            {
                sum_val += ::exp(val-max_val);
            }
        }
        // Reduce max
        volatile __shared__ T block_max_val;
        __shared__ T block_sum_val;
        if(i0_start == 0)
        {
            block_max_val = max_val;
            block_sum_val = zero;
        }
        __syncthreads();
        while(block_max_val < max_val)
        {
            block_max_val = max_val;
        }
        __syncthreads();
        // Update own sum
        sum_val *= ::exp(max_val-block_max_val);
        atomicAdd(&block_sum_val, sum_val);
        // Save result
        __syncthreads();
        if(i0_start == 0)
        {
            Index dst_offset = i1 + i2*m;
            T &max_output = maxsumexp[2*dst_offset];
            T &sum_output = maxsumexp[2*dst_offset+1];
            if(sum_output == zero)
            {
                max_output = block_max_val;
                sum_output = block_sum_val;
            }
            else
            {
                if(block_max_val < max_output)
                {
                    block_sum_val *= ::exp(block_max_val-max_output);
                }
                else
                {
                    sum_output *= ::exp(max_output-block_max_val);
                    max_output = block_max_val;
                }
                sum_output += block_sum_val;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
        T *maxsumexp)
    noexcept
//! Max and sum of exponents along middle axis
/*! For a provided m-by-k-by-n input array src compute maximums and sums of
 * exponents of slices along second axis with k elements, resulting in
 * 2-by-m-by-n output array maxsumexp.
 *
 *      old[0,i,j] = maxsumexp[0,i,j]
 *      old[1,i,j] = maxsumexp[1,i,j]
 *      maxsumexp[0,i,j] = max(old[0,i,j], max(src[i,:,j]))
 *      maxsumexp[1,i,j] = old[1,i,j]*exp(old[0,i,j]-maxsumexp[0,i,j])
 *          + sum(exp(src[i,:,j]-maxsumexp[0,i,j])))
 *
 * @param[in] m: Size of the first mode of src and the second mode of maxsumexp
 *      arrays.
 * @param[in] n: Size of the last mode of src and maxsumexp arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array, that accumulates
 *      sums and norms of slices along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 1), std::min(int(n), 1), 64);
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, src,
            maxsumexp);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *maxsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *maxsumexp)
    noexcept;

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

