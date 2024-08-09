/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_backward/cuda.cu
 * Backward SiLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::silu_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *x, const T *dy, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y one{1.0};
    Y x_val{0.0};
    Y dy_val{0.0};
    Y dx_val{0.0};
    Y sigma{0.0};
    if(i < nelems)
    {
        x_val = Y{x[i]};
        dy_val = Y{dy[i]};
        dx_val = Y{dx[i]};
        sigma = one / (one + ::exp(-x_val));
        dx[i] = T{dx_val + dy_val * sigma * (one + x_val * (one - sigma))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x_, const T *dy_, T *dx_)
    noexcept
//! Backward SiLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*SiLU'(x[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward SiLU
 * @params[in] dy: Gradient over output of forward SiLU
 * @params[inout] dx: Gradient over input of forward SiLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, x_, dy_, dx_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *x,
        const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index nelems, const fp32_fast_tf32_t *x,
        const fp32_fast_tf32_t *dy, fp32_fast_tf32_t *dx)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *x,
        const fp64_t *dy, fp64_t *dx)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *x,
        const bf16_t *dy, bf16_t *dx)
    noexcept;

} // namespace nntile::kernel::silu_backward
