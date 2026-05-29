/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T * __restrict__ src1, Scalar beta_, const T * __restrict__ src2,
        T * __restrict__ dst)
//! Generic implementation of the add_slice operation on CUDA
/*! @copydoc nntile::kernel::add_slice::cuda
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero = 0.0;
    const Y beta = beta_;
    const Y alpha = alpha_;
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the input array src2
        const T *src2_fiber = src2 + i1*mk + i0;
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to add to the output fiber
        const Y src1_val = alpha * Y{src1[i1*m+i0]};
        // And update it
        dst_fiber[i2*m] = T{beta * Y{src2_fiber[i2*m]} + src1_val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept
//! Add a tensor and a broadcasted slice with optional scaling on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = alpha*src1[i,j] + beta*src2[i,l,j]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i,j] is NaN, then dst[i,l,j] will be NaN.
 * If beta is zero and src2[i,l,j] is NaN, then dst[i,l,j] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_slice instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_slice
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src1, src2 and dst tensors
 * @param[in] n: Size of the last mode of src1, src2 and dst tensors
 * @param[in] k: Size of the middle mode of src2 and dst tensor
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src1: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for src1
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
            src1, beta, src2, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_slice
