/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cuda.cc
 * Norm of a buffer on GPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */


#include "nntile/kernel/norm/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace norm
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T beta, T *norm_dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0;
    // Cycle over row of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over column of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Get norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            Index dst_offset = i1 + i2*m;
            // Init norm
            T norm_max = zero, norm_ssq = zero;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
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
            norm_max *= alpha;
            T norm = norm_max * std::sqrt(norm_ssq);
            // Save result
            if(beta == zero)
            {
                norm_dst[dst_offset] = norm;
            }
            else
            {
                norm_dst[dst_offset] = std::hypot(beta*norm_dst[dst_offset],
                        norm);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *norm_dst)
    noexcept
//! Norm along middle axis
/*! For a provided m-by-k-by-n input array src compute norms of slices
 * along second axis with k elements, resulting in m-by-n output array
 * norm_dst. Mnemonically, the following operations are performed:
 *      norm_dst[i,j] = hypot(beta*norm_dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and norm_dst arrays
 * @param[in] n: Size of the last mode of src and norm_dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for norm_dst
 * @param[inout] norm_dst: Output contiguous m-by-n array, that accumulates
 *      norms along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha, src,
            beta, norm_dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *norm_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *norm_dst)
    noexcept;

} // namespace norm
} // namespace kernel
} // namespace nntile

