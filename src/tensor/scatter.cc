/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scatter.cc
 * Scatter operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scatter.hh"
#include <algorithm>
#include "nntile/tile/copy_intersection.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise scatter operation
/*! Scatter a single-tiled tensor, stored on a single node, into a distributed
 * grid of tiles.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void scatter_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check if source is a single-tile tensor
    if(src.grid.nelems != 1)
    {
        throw std::runtime_error("Source must be a single-tiled tensor");
    }
    // Check if shapes match
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    Index ndim = src.ndim;
    tile::Tile<int64_t> scratch_tile({std::max<Index>(1, 2*ndim)});
    auto src_tile = src.get_tile(0);
    std::vector<Index> src_offset(ndim, 0);
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto dst_tile = dst.get_tile(i);
        auto dst_tile_index = dst.grid.linear_to_index(i);
        std::vector<Index> dst_offset(ndim);
        for(Index k = 0; k < ndim; ++k)
        {
            dst_offset[k] = dst_tile_index[k] * dst.basetile_shape[k];
        }
        tile::copy_intersection_async<T>(src_tile, src_offset, dst_tile,
                dst_offset, scratch_tile);
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise scatter operation
/*! Scatter a single-tiled tensor, stored on a single node, into a distributed
 * grid of tiles.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void scatter(const Tensor<T> &src, const Tensor<T> &dst)
{
    scatter_async<T>(src, dst);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scatter_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void scatter_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void scatter_async<int64_t>(const Tensor<int64_t> &src,
        const Tensor<int64_t> &dst);

template
void scatter_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void scatter_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void scatter_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
                               const Tensor<fp32_fast_bf16_t> &dst);

template
void scatter_async<bool_t>(const Tensor<bool_t> &src,
        const Tensor<bool_t> &dst);

template
void scatter_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void scatter_async<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void scatter<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void scatter<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
                               const Tensor<fp32_fast_tf32_t> &dst);

template
void scatter<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
                               const Tensor<fp32_fast_fp16_t> &dst);

template
void scatter<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
                               const Tensor<fp32_fast_bf16_t> &dst);

template
void scatter<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void scatter<int64_t>(const Tensor<int64_t> &src, const Tensor<int64_t> &dst);

template
void scatter<bool_t>(const Tensor<bool_t> &src, const Tensor<bool_t> &dst);

template
void scatter<bf16_t>(const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

template
void scatter<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
