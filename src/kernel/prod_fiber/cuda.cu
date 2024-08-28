/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_fiber/cuda.cu
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/prod_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::prod_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, T alpha, const T *src, T *dst)
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
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i0 < m and i1 < n and i2 < k)
    {
        const T src_val = alpha * src[i2];
        // Output fiber to be updated
        T *dst_fiber = dst + (i1*k+i2)*m;
        // Update output value
        dst_fiber[i0] *= src_val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src_, T *dst_)
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
    using Y = typename CUDAComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto dst = reinterpret_cast<Y *>(dst_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(m, n, k, Y{alpha}, src,
            dst);
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

} // namespace nntile::kernel::prod_fiber
