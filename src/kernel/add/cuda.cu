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
void cuda_kernel(Index nelems, Scalar alpha_, const T *src1, const T *src2, Scalar beta_, T *dst)

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
	using Y = typename T::repr_t;
        Y src1_val = Y{src1[i]};
	Y src2_val = Y{src2[i]};
        Y dst_val = Y{dst[i]};
        Y alpha = Y{alpha_};
        Y beta = Y{beta_};
        dst[i] = alpha * src1_val + beta * src2_val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src1_, const T *src2_, Scalar beta_, T *dst_)
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
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(nelems, alpha_, src1_, src2_, beta_, dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t *src1, const fp32_t *src2, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t *src1, const fp64_t *src2, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t *src1, const bf16_t *src2, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add
