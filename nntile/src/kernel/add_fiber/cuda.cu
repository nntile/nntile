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
void cuda_kernel(
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha_,
    const T *src1,
    Scalar beta_,
    const T *src2,
    T *dst
)
//! Generic implementation of the add_fiber operation on CUDA
/*! @copydoc nntile::kernel::add_fiber::cuda
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
            const Y src1_val = alpha * static_cast<Y>(src1[i2+b*k]);
	        const T *src2_fiber = src2 + ((i1+b*n)*k+i2)*m;
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            const Y src2_val = static_cast<Y>(src2_fiber[i0]);
            // And update output
            dst_fiber[i0] = static_cast<T>(beta * src2_val + src1_val);
        }
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add a tensor and a broadcasted fiber with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and src2[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of dst tensor.
 * @param[in] n: Size of the last mode of dst tensor.
 * @param[in] k: Size of the middle mode of dst and src2 tensors, and the first
 * mode of src1.
 * @param[in] batch: Size of the batch dimension.
 * @param[in] alpha: Scalar factor for src1.
 * @param[in] src1: Input contiguous vector with k*batch elements.
 * @param[in] beta: Scaling factor for src2.
 * @param[in] src2: Input contiguous tensor with m*k*n*batch elements.
 * @param[inout] dst: Output contiguous m-by-k-by-n-by-batch tensor.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src1, beta, src2, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src1, Scalar beta,
	const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src1, Scalar beta,
	const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src1, Scalar beta,
	const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src1, Scalar beta,
	const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
