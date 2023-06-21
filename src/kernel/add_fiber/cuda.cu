/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#include "nntile/kernel/add_fiber/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace add_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, T alpha, const T *src, T beta,
        T *dst)
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    constexpr T zero = 0;
    if(i2 < k and i1 < n and i0 < m)
    {
        // Value to add to the output slice
        const T src_val = alpha * src[i2];
        // Output fiber to be updated
        T *dst_fiber = dst + (i1*k+i2)*m;
        // Overwrite or update output depending on beta
        if(beta == zero)
        {
                // Set output value
                dst_fiber[i0] = src_val;
        }
        else
        {
                // Read value from the output
                T &dst_val = dst_fiber[i0];
                // And update it
                dst_val = beta*dst_val + src_val;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src, beta,
            dst);
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

} // namespace add_fiber
} // namespace kernel
} // namespace nntile

