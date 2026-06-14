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
void cuda_kernel(Index nrows, Index ncols, const T *sin, const T *cos,
    const T *src, T *dst)
{
    const Index m = ncols;
    const Index n = nrows;
    Index flat = threadIdx.x + blockIdx.x * blockDim.x;
    if(flat < m * n)
    {
        using Y = typename T::repr_t;
        const Index j = flat / m;
        const Index i = flat % m;
        const Index l = 2 * (i + j * m);
        Y c{cos[i]}, s{sin[i]};
        Y a{src[l]}, b{src[l + 1]};
        dst[l] = static_cast<T>(c * a - s * b);
        dst[l + 1] = static_cast<T>(s * a + c * b);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nrows, Index ncols, const T *sin,
    const T *cos, const T *src, T *dst) noexcept
{
    const Index m = ncols;
    const Index n = nrows;
    dim3 blocks((m * n + 255) / 256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(nrows, ncols, sin, cos,
        src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nrows, Index ncols,
    const fp32_t *sin, const fp32_t *cos, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nrows, Index ncols,
    const fp64_t *sin, const fp64_t *cos, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nrows, Index ncols,
    const fp16_t *sin, const fp16_t *cos, const fp16_t *src, fp16_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nrows, Index ncols,
    const bf16_t *sin, const bf16_t *cos, const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::rope
