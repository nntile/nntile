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
 * @version 1.1.0
 * */

#include "nntile/kernel/rope/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::rope
{

template<typename T>
static __global__
void cuda_kernel(Index m_pairs, Index n, Index sin_pair0, const T *sin,
    const T *cos, const T *src, T *dst)
/*! Apply RoPE to src and write into dst (C-order head layout).
 *
 * @param[in] m_pairs: Number of head pairs
 * @param[in] n: Spatial extent per pair
 * @param[in] sin_pair0: Offset of the first sin/cos pair in sin/cos buffers
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    Index flat = threadIdx.x + blockIdx.x * blockDim.x;
    if(flat < m_pairs * n)
    {
        using Y = typename T::repr_t;
        const Index j = flat % n;
        const Index i = flat / n;
        const Index si = sin_pair0 + i * n + j;
        const Index l0 = 2 * i * n + j;
        const Index l1 = l0 + n;
        Y c{cos[si]}, s{sin[si]};
        Y a{src[l0]}, b{src[l1]};
        dst[l0] = static_cast<T>(c*a - s*b);
        dst[l1] = static_cast<T>(s*a + c*b);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m_pairs, Index n, Index m_sin,
    Index sin_pair0, const T *sin, const T *cos, const T *src, T *dst)
    noexcept
{
    (void)m_sin;
    dim3 blocks((m_pairs*n+255)/256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(m_pairs, n, sin_pair0, sin,
        cos, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m_pairs, Index n, Index m_sin,
    Index sin_pair0, const fp32_t *sin, const fp32_t *cos, const fp32_t *src,
    fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m_pairs, Index n, Index m_sin,
    Index sin_pair0, const fp64_t *sin, const fp64_t *cos, const fp64_t *src,
    fp64_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m_pairs, Index n, Index m_sin,
    Index sin_pair0, const fp16_t *sin, const fp16_t *cos, const fp16_t *src,
    fp16_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m_pairs, Index n, Index m_sin,
    Index sin_pair0, const bf16_t *sin, const bf16_t *cos, const bf16_t *src,
    bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::rope
