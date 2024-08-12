/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh/cuda.cu
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    // Constants
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        Y z = Y{src[i]};
        Y y = z * (f3 + f4*z*z);
        dst[i] = T{z / (one + ::exp(y))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Approximate GeLU operation on CUDA
/*! Applies the following approximation of the GeLU function:
 * GeLU(z) \approx 0.5z(1+tanh(sqrt(2/pi)(z+0.044715z^3))),
 * which is actually implemented as
 * GeLU(z) \approx z/(1+exp(-2sqrt(2/pi)z(1+0.044715z^2)))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input buffer to apply GeLU
 * @params[out] dst: Output buffer to apply GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t * src,
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

} // namespace nntile::kernel::gelutanh
