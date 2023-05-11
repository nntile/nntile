/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_slice/cuda.cu
 * Euclidean norms of fibers into a slice of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#include "nntile/kernel/norm_slice/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace norm_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T beta, T *dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0, one = 1.0;
    // Cycle over column of the output buffer dst
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init norm of the fiber
            T norm_max = zero, norm_ssq = zero;
            // Output value
            T &result = dst[i2*m+i1];
            // Cycle over fiber elements and accumulate the norm
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = abs(src_fiber[i0*m]);
                // Update norm only if new value is non-zero
                if(val > 0)
                {
                    if(norm_max >= val)
                    {
                        T tmp1 = val / norm_max;
                        norm_ssq += tmp1 * tmp1;
                    }
                    else
                    {
                        T tmp1 = norm_max / val;
                        T tmp2 = tmp1 * tmp1;
                        norm_ssq = one + norm_ssq*tmp2;
                        norm_max = val;
                    }
                }
            }
            // Get the scaled norm
            norm_max *= alpha;
            T norm_slice = norm_max * std::sqrt(norm_ssq);
            // Update output value
            if(beta == zero)
            {
                result = norm_slice;
            }
            else
            {
                result = std::hypot(beta*result, norm_slice);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-n array, that
 *      accumulates norms along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha, src,
            beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace norm_slice
} // namespace kernel
} // namespace nntile

