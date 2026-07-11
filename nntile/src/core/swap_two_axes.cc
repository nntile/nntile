/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/core/swap_two_axes.cc
 * swap_two_axes operation for Tile<T>.
 *
 * @version 1.1.0
 * */

#include "nntile/core/swap_two_axes.hh"

#include "nntile/core/swap_two_axes_decompose.hh"
#include "nntile/starpu/config.hh"
#include "nntile/starpu/swap_two_axes.hh"

namespace nntile::core
{

namespace
{

std::vector<Index> tile_shape_vector(const TileTraits &traits)
{
    return traits.shape;
}

} // namespace

template<typename T>
void swap_two_axes_async(
    int starpu_worker_hint,
    const Tile<T> &src,
    const Tile<T> &dst,
    Index dim0,
    Index dim1)
{
    const std::vector<Index> src_shape = tile_shape_vector(src);
    const SwapTwoAxesDecomposition decomp =
        decompose_swap_axes(src_shape, dim0, dim1);
    const auto &d = decomp.sizes_5d;
    const std::vector<Index> &dst_shape = decomp.output_shape;
    if (dst_shape != tile_shape_vector(dst))
    {
        throw std::runtime_error("swap_two_axes: dst shape mismatch");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if (mpi_rank == dst_rank)
    {
        starpu::swap_two_axes.submit<std::tuple<T>>(
            starpu_worker_hint,
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            src,
            dst);
    }
}

template<typename T>
void swap_two_axes(
    int starpu_worker_hint,
    const Tile<T> &src,
    const Tile<T> &dst,
    Index dim0,
    Index dim1)
{
    swap_two_axes_async<T>(starpu_worker_hint, src, dst, dim0, dim1);
    nntile::starpu_task_wait_for_all_unless_deferred();
}

template
void swap_two_axes_async<fp32_t>(
    int starpu_worker_hint,
    const Tile<fp32_t> &src,
    const Tile<fp32_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp32_t>(
    int starpu_worker_hint,
    const Tile<fp32_t> &src,
    const Tile<fp32_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<fp64_t>(
    int starpu_worker_hint,
    const Tile<fp64_t> &src,
    const Tile<fp64_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp64_t>(
    int starpu_worker_hint,
    const Tile<fp64_t> &src,
    const Tile<fp64_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<bf16_t>(
    int starpu_worker_hint,
    const Tile<bf16_t> &src,
    const Tile<bf16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<bf16_t>(
    int starpu_worker_hint,
    const Tile<bf16_t> &src,
    const Tile<bf16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<fp16_t>(
    int starpu_worker_hint,
    const Tile<fp16_t> &src,
    const Tile<fp16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp16_t>(
    int starpu_worker_hint,
    const Tile<fp16_t> &src,
    const Tile<fp16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<fp32_fast_tf32_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_tf32_t> &src,
    const Tile<fp32_fast_tf32_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp32_fast_tf32_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_tf32_t> &src,
    const Tile<fp32_fast_tf32_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<fp32_fast_fp16_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_fp16_t> &src,
    const Tile<fp32_fast_fp16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp32_fast_fp16_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_fp16_t> &src,
    const Tile<fp32_fast_fp16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes_async<fp32_fast_bf16_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_bf16_t> &src,
    const Tile<fp32_fast_bf16_t> &dst,
    Index dim0,
    Index dim1);

template
void swap_two_axes<fp32_fast_bf16_t>(
    int starpu_worker_hint,
    const Tile<fp32_fast_bf16_t> &src,
    const Tile<fp32_fast_bf16_t> &dst,
    Index dim0,
    Index dim1);

} // namespace nntile::core
