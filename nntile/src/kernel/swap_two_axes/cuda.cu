/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/kernel/swap_two_axes/cuda.cu
 * Swap axes 1 and 3 in a 5D buffer on CUDA.
 *
 * Layout is ``[d0, d1, d2, d3, d4]`` -> ``[d0, d3, d2, d1, d4]``.
 *
 * - ``d4 > 1`` (non-fastest axis swap): each warp copies a contiguous fiber
 *   ``src[i0,i1,i2,i3,:]`` -> ``dst[i0,i3,i2,i1,:]``, using ``float4`` when
 *   ``d4`` is a multiple of the vector width.
 * - ``d4 == 1`` (fastest-axis involvement, e.g. HF ``transpose(-1,-2)``):
 *   for each ``(i0,i2)`` this is a ``d1xd3`` matrix transpose. A separate
 *   tiled shared-memory kernel coalesces both loads and stores.
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
//! Warps per block for the contiguous-fiber (d4 > 1) kernel.
constexpr int warps_per_block = 8;

//! Shared-memory tile for the d4 == 1 matrix-transpose kernel.
constexpr int tile_dim = 32;
constexpr int block_rows = 8;

} // namespace

//! Contiguous-fiber copy for d4 > 1 (one warp per fiber).
template<typename T>
static __global__
void cuda_kernel_fiber(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    Index n_fibers,
    const T * __restrict__ src,
    T * __restrict__ dst)
{
    const Index fiber =
        static_cast<Index>(blockIdx.x) * warps_per_block +
        static_cast<Index>(threadIdx.y);
    if (fiber >= n_fibers)
    {
        return;
    }

    Index rem = fiber;
    const Index i3 = rem % d3;
    rem /= d3;
    const Index i2 = rem % d2;
    rem /= d2;
    const Index i1 = rem % d1;
    const Index i0 = rem / d1;

    const Index src_base =
        ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4);
    const Index dst_base =
        ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4);

    constexpr Index vec_width =
        static_cast<Index>(sizeof(float4) / sizeof(T));
    if constexpr (vec_width >= 1)
    {
        if ((d4 % vec_width) == 0)
        {
            const Index n_vec = d4 / vec_width;
            const float4 *src_v = reinterpret_cast<const float4 *>(
                src + src_base);
            float4 *dst_v = reinterpret_cast<float4 *>(dst + dst_base);
            for (Index i = static_cast<Index>(threadIdx.x); i < n_vec;
                i += warp_size)
            {
                dst_v[i] = src_v[i];
            }
            return;
        }
    }

    for (Index i4 = static_cast<Index>(threadIdx.x); i4 < d4; i4 += warp_size)
    {
        dst[dst_base + i4] = src[src_base + i4];
    }
}

/*! Tiled transpose of the (d1 x d3) plane for each (i0, i2) when d4 == 1.
 *
 * Src element ``(i1,i3)`` is contiguous in ``i3``; dst element ``(i3,i1)`` is
 * contiguous in ``i1``. Shared-memory tiling coalesces both sides (NVIDIA
 * matrix-transpose pattern, padded tile to avoid bank conflicts).
 */
template<typename T>
static __global__
void cuda_kernel_d4_eq_1(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index n_tiles_d3,
    Index n_tiles_d1,
    const T * __restrict__ src,
    T * __restrict__ dst)
{
    __shared__ T tile[tile_dim][tile_dim + 1];

    const Index n_tiles_plane = n_tiles_d3 * n_tiles_d1;
    const Index batch = static_cast<Index>(blockIdx.x) / n_tiles_plane;
    const Index tile_id = static_cast<Index>(blockIdx.x) % n_tiles_plane;
    const Index tile_i1 = tile_id / n_tiles_d3;
    const Index tile_i3 = tile_id % n_tiles_d3;

    const Index i0 = batch / d2;
    const Index i2 = batch % d2;

    const Index src_row_stride = d2 * d3;
    const Index src_plane_base = (i0 * d1 * d2 + i2) * d3;
    const Index dst_row_stride = d2 * d1;
    const Index dst_plane_base = (i0 * d3 * d2 + i2) * d1;

    const Index tx = static_cast<Index>(threadIdx.x);
    const Index ty = static_cast<Index>(threadIdx.y);
    const Index row0 = tile_i1 * tile_dim;
    const Index col0 = tile_i3 * tile_dim;

    // Coalesced load: warp reads consecutive i3 into a shared-memory row.
    for (Index j = 0; j < tile_dim; j += block_rows)
    {
        const Index i1 = row0 + ty + j;
        const Index i3 = col0 + tx;
        if (i1 < d1 && i3 < d3)
        {
            tile[ty + j][tx] =
                src[src_plane_base + i1 * src_row_stride + i3];
        }
    }
    __syncthreads();

    // Coalesced store: warp writes consecutive i1 from a shared-memory column.
    for (Index j = 0; j < tile_dim; j += block_rows)
    {
        const Index i3 = col0 + ty + j;
        const Index i1 = row0 + tx;
        if (i3 < d3 && i1 < d1)
        {
            dst[dst_plane_base + i3 * dst_row_stride + i1] =
                tile[tx][ty + j];
        }
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
    if (d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0 || d4 <= 0)
    {
        return;
    }

    if (d4 == 1)
    {
        const Index n_tiles_d3 = (d3 + tile_dim - 1) / tile_dim;
        const Index n_tiles_d1 = (d1 + tile_dim - 1) / tile_dim;
        const Index n_batch = d0 * d2;
        const Index n_blocks = n_tiles_d3 * n_tiles_d1 * n_batch;
        dim3 blocks(static_cast<unsigned>(n_blocks));
        dim3 threads(tile_dim, block_rows);
        (cuda_kernel_d4_eq_1<T>)<<<blocks, threads, 0, stream>>>(
            d0,
            d1,
            d2,
            d3,
            n_tiles_d3,
            n_tiles_d1,
            src,
            dst);
        return;
    }

    const Index n_fibers = d0 * d1 * d2 * d3;
    const Index n_blocks =
        (n_fibers + warps_per_block - 1) / warps_per_block;
    dim3 blocks(static_cast<unsigned>(n_blocks));
    dim3 threads(warp_size, warps_per_block);
    (cuda_kernel_fiber<T>)<<<blocks, threads, 0, stream>>>(
        d0,
        d1,
        d2,
        d3,
        d4,
        n_fibers,
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
