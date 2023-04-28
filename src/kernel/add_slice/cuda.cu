/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#include "nntile/kernel/add_slice/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace add_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0;
    // Cycle over column of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over row of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Pointer to a corresponding fiber of the output array dst
            T *dst_fiber = dst + i2*mk + i1;
            // Value to add to the output fiber
            const T src_val = alpha * src[i2*m+i1];
            // Overwrite or update output depending on beta
            if(beta == zero)
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Set output value
                    dst_fiber[i0*m] = src_val;
                }
            }
            else
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Read value from the output
                    T &dst_val = dst_fiber[i0*m];
                    // And update it
                    dst_val = beta*dst_val + src_val;
                }
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted slice on CUDA
/*! This is a host function that does the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
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

} // namespace add_slice
} // namespace kernel
} // namespace nntile

