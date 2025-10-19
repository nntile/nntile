/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_fiber_inplace/cuda.cu
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/prod_fiber_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::prod_fiber_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Scalar alpha_, const T *src, T *dst)
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input contiguous vector with k elements
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i0 < m and i1 < n and i2 < k)
    {
        const Y src_val = alpha * Y{src[i2]};
        // Output fiber to be updated
        T *dst_fiber = dst + (i1*k+i2)*m;
        // Update output value
        dst_fiber[i0] = static_cast<T>(src_val * Y{dst_fiber[i0]});
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, T *dst)
    noexcept
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src_: Input contiguous vector with k elements
 * @param[inout] dst_: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::prod_fiber_inplace
