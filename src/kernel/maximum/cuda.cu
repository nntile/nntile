/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maximum/cuda.cu
 * Maximum operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/maximum/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::maximum
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T* src, T* dst)
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = T{::fmax(static_cast<Y>(dst[i]), static_cast<Y>(src[i]))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Inplace maximum operation on CUDA
/*! Does the following per-element operation:
 * dst[i] := max(src[i], dst[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: input buffer
 * @params[inout] dst_: buffer for comparison and store maximum
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t* src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t* src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t* src,
        bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::maximum
