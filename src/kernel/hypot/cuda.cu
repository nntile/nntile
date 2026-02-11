/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot/cuda.cu
 * hypot operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/hypot/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::hypot
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T* src1, Scalar beta, const T* src2, T* dst)
//! Generic implementation of the hypot operation on CUDA
/*! @copydoc nntile::kernel::hypot::cuda
 * */
{
    Index i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha};
    const Y beta_{beta};
    if(i < nelems)
    {
        const Y src1_val = static_cast<Y>(src1[i]);
        const Y src2_val = static_cast<Y>(src2[i]);
        dst[i] = static_cast<T>(std::hypot(alpha*src1_val, beta*src2_val));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T* src1, Scalar beta, const T* src2, T* dst)
    noexcept
//! Hypothenuse of two buffers with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i] = hypot(alpha*src1[i], beta*src2[i])
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and src2[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha or beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination tensor
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src1, beta, src2, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t* src1,
        Scalar beta, const fp32_t* src2, fp32_t* dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t* src1,
        Scalar beta, const fp64_t* src2, fp64_t* dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t* src1,
        Scalar beta, const bf16_t* src2, bf16_t* dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t* src1,
        Scalar beta, const fp16_t* src2, fp16_t* dst)
    noexcept;

} // namespace nntile::kernel::hypot
