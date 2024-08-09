/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cuda.cu
 * Backward approximate GeLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *x, const T *dy, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    using Y = typename T::repr_t;
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1, f5 = Y{3}*f4;
    if(i < nelems)
    {
        Y z = Y{x[i]};
        Y z2 = z * z;
        Y y1 = z * (f3 + f4*z2);
        Y y2 = z * (f3 + f5*z2);
        Y expy1 = exp(y1);
        if(not isinf(expy1))
        {
            Y inv_expy1p1 = one / (expy1 + one);
            Y dx_val = Y{dx[i]};
            dx[i] = T{dx_val + (one-y2*(one-inv_expy1p1)) * inv_expy1p1 * Y{dy[i]}};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x, const T *dy, T *dx)
    noexcept
//! Backward approximate GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*GeLUtanh'(x[i])
 * GeLUtanh'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward GeLU
 * @params[in] dy: Gradient over output of forward GeLU
 * @params[inout] dx: Gradient over input of forward GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, x, dy, dx);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *x,
        const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *x,
        const fp64_t *dy, fp64_t *dx)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *x,
        const bf16_t *dy, bf16_t *dx)
    noexcept;

} // namespace nntile::kernel::gelutanh_backward
