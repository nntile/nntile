/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scal/cuda.cu
 * Scal operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scal/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::scal
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T* src, T* dst)
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[out] dst_: Destination of the scal operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i < nelems)
    {
        dst[i] = alpha * Y{src[i]};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src_,
        T *dst_)
    noexcept
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[out] dst_: Destination of the scal operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src_,
            dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::scal
