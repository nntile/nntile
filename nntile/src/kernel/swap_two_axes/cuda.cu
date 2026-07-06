/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/kernel/swap_two_axes/cuda.cu
 * Swap axes 1 and 3 in a 5D buffer on CUDA.
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/swap_two_axes/cuda.hh"

#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::swap_two_axes
{

namespace
{

constexpr int warp_size = 32;

} // namespace

template<typename T>
static __global__
void cuda_kernel(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *src,
    T *dst)
{
    const Index fiber = blockIdx.x;
    Index rem = fiber;
    const Index i3 = rem % d3;
    rem /= d3;
    const Index i2 = rem % d2;
    rem /= d2;
    const Index i1 = rem % d1;
    const Index i0 = rem / d1;
    if (i0 >= d0)
    {
        return;
    }
    const Index src_base =
        ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4);
    const Index dst_base =
        ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4);
    for (Index i4 = threadIdx.x; i4 < d4; i4 += warp_size)
    {
        dst[dst_base + i4] = src[src_base + i4];
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *src,
    T *dst) noexcept
{
    const Index n_fibers = d0 * d1 * d2 * d3;
    dim3 blocks(static_cast<unsigned>(n_fibers));
    dim3 threads(warp_size);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
        d0,
        d1,
        d2,
        d3,
        d4,
        src,
        dst);
}

template
void cuda<fp32_t>(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp32_t *src,
    fp32_t *dst) noexcept;

template
void cuda<fp64_t>(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp64_t *src,
    fp64_t *dst) noexcept;

template
void cuda<bf16_t>(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const bf16_t *src,
    bf16_t *dst) noexcept;

template
void cuda<fp16_t>(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp16_t *src,
    fp16_t *dst) noexcept;

} // namespace nntile::kernel::swap_two_axes
