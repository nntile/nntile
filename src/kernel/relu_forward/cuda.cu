/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu_forward/cuda.cu
 * Forward ReLU operation on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/relu_forward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::relu_forward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src_, T *dst_)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::compat_t;
    using Z = typename CUDAComputeType<T>::value;
    const Z* src = reinterpret_cast<const Z *>(src_);
    Z* dst = reinterpret_cast<Z *>(dst_);
    constexpr Y zero = Y{0.0};
    Y src_val{0.0};
    if(i < nelems)
    {   
        src_val = Y{src[i]};
        dst[i] = Z{::fmax(src_val, zero)};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src_, T *dst_)
    noexcept
//! Forward ReLU operation on CUDA
/*! Does the following per-element operation:
 * dst[i] = max(src[i], 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input array
 * @params[out] dst_: Output array
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    // using Y = typename CUDAComputeType<T>::value;
    // auto src = reinterpret_cast<const Y *>(src_);
    // auto dst = reinterpret_cast<Y *>(dst_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src_, dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *src,
        bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::relu_forward
