/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j,b] = beta*dst[i,l,j,b] + alpha*src[l,b]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to add to the output slice
            const Y src_val = alpha * Y{src[i2+b*k]};
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            T &dst_val = dst_fiber[i0];
            // And update it
            dst_val = T{beta * Y{dst_val} + src_val};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src, beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
