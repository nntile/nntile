/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/add/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        using Y = typename T::compat_t;
        using Z = typename CUDAComputeType<T>::value;
        auto src_ = reinterpret_cast<const Z *>(src);
        auto dst_ = reinterpret_cast<Z *>(dst);

        Y src_val = Y{src_[i]};
        Y dst_val = Y{dst_[i]};
        Y alpha = Y{alpha_};
        Y beta = Y{beta_};
        dst_[i] = Z{alpha * src_val + beta * dst_val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src_,
        Scalar beta_, T *dst_)
    noexcept
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add operation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    // auto src = cast_pointer_cuda<T>(src_);
    // auto dst = cast_pointer_cuda<T>(dst_);
    // auto alpha = cast_scalar_cuda<T>(alpha_);
    // auto beta = cast_scalar_cuda<T>(beta_);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(nelems, alpha_, src_,
            beta_, dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add
