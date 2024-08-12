/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_forward/cuda.cu
 * Forward SiLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_forward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::silu_forward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y one = Y{1.0};
    Y src_val{0.0};
    if(i < nelems)
    {
        src_val = Y{src[i]};
        dst[i] = T{src_val / (one + ::exp(-src_val))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src_, T *dst_)
    noexcept
//! Forward SiLU operation on CUDA
/*! Does the following per-element operation:
 * dst[i] = src[i] * sigmoid(src[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input array
 * @params[out] dst_: Output array
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src_, dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index nelems, const fp32_fast_tf32_t *src,
        fp32_fast_tf32_t *dst)
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
