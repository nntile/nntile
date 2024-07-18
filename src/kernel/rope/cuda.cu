/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/rope/cuda.cu
 * Rotary Positional Embedding
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/rope/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::rope
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, const T *sin, const T *cos, const T *src,
        T *dst)
/*! Change provided 2-by-m-by-n src tensor and write result into dst tensor
 *  sin, cos are tensors of shape (m). Each column holds sines and cosines.
 *  dst[2i,j] = cos[i] * src[2i,j] - sin[i] * src[2i+1,j]
 *  dst[2i+1,j] = sin[i] * src[2i,j] + cos[i] * src[2i+1,j]
 *
 * @param[in] m: Size of sin and cos tensors
 * @param[in] n: Size of the second mode of src and dst tensors
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < m*n)
    {
        using Y = typename T::repr_t;
        int j = i % m;
        Y c{cos[i]}, s{sin[i]};
        Y a{src[2*i]}, b{src[2*i+1]};
        dst[2*i] = static_cast<T>(c*a - s*b);
        dst[2*i+1] = static_cast<T>(s*a + c*b);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, const T *sin, const T *cos,
        const T *src, T *dst)
    noexcept
{
    dim3 blocks((m*n+255)/256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(m, n, sin, cos, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, const T *sin,
        const T *cos, const T *src, T *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, const T *sin,
        const T *cos, const T *src, T *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n,
        const T *sin, const T *cos, const T *src, T *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, const T *sin,
        const T *cos, const T *src, T *dst)
    noexcept;

} // namespace nntile::kernel::rope
