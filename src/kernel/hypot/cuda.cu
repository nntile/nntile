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
void cuda_kernel(Index nelems, Scalar alpha_, const T* src1, Scalar beta_, const T* src2, T* dst)
//! hypot two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src1[i], beta*src2[i]),
 * which computes sqrt((alpha*src1[i])^2 + (beta*src2[i])^2).
 *
 * @param[in] nelems: Size of the src1, src2 and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta_: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination of the hypot operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};
    Y alpha{alpha_};
    Y beta{beta_};
    if(i < nelems)
    {
        if(alpha == zero)
        {
            if(beta == zero)
            {
                dst[i] = zero;
            }
            else
            {
                Y src2_val = static_cast<Y>(src2[i]);
                dst[i] = ::fabs(beta * src2_val);
            }
        }
        else
        {
            if(beta == zero)
            {
                Y src1_val = static_cast<Y>(src1[i]);
                dst[i] = ::fabs(alpha * src1_val);
            }
            else
            {
                Y src1_val = static_cast<Y>(src1[i]);
                Y src2_val = static_cast<Y>(src2[i]);
                dst[i] = ::hypot(alpha*src1_val, beta*src2_val);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T* src1, Scalar beta, const T* src2, T* dst)
    noexcept
//! hypot two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src1[i], beta*src2[i]),
 * which computes sqrt((alpha*src1[i])^2 + (beta*src2[i])^2).
 *
 * @param[in] nelems: Size of the src1, src2 and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination of the hypot operation
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
