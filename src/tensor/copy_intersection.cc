/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy_intersection.cc
 * Copy intersection of 2 tensors from one into another
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/copy_intersection.hh"
#include "nntile/tile/copy_intersection.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated. Both the
 * source and the target tensors assumed to have the same offset.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * @param[scratch] scratch: Temporary workspace
 * */
template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    // Check dimensions
    if(src.ndim != src_offset.size())
    {
        throw std::runtime_error("src.ndim != src_offset.size()");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_offset.size())
    {
        throw std::runtime_error("dst.ndim != dst_offset.size()");
    }
    Index ndim = src.ndim;
    tile::Tile<int64_t> scratch_tile({std::max<Index>(1, 2*ndim)});
    for(Index dst_linear = 0; dst_linear < dst.grid.nelems; ++dst_linear)
    {
        auto dst_tile = dst.get_tile(dst_linear);
        auto dst_tile_handle = dst.get_tile_handle(dst_linear);
        auto dst_tile_index = dst.grid.linear_to_index(dst_linear);
        std::vector<Index> dst_tile_offset(ndim);
        for(Index d = 0; d < ndim; ++d)
        {
            dst_tile_offset[d] = dst_offset[d]
                + dst_tile_index[d] * dst.basetile_shape[d];
        }
        for(Index src_linear = 0; src_linear < src.grid.nelems; ++src_linear)
        {
            auto src_tile = src.get_tile(src_linear);
            auto src_tile_index = src.grid.linear_to_index(src_linear);
            std::vector<Index> src_tile_offset(ndim);
            for(Index d = 0; d < ndim; ++d)
            {
                src_tile_offset[d] = src_offset[d]
                    + src_tile_index[d] * src.basetile_shape[d];
            }
            tile::copy_intersection_async<T>(src_tile, src_tile_offset, dst_tile,
                    dst_tile_offset, scratch_tile);
        }
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * @param[scratch] scratch: Temporary workspace
 * */
template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void copy_intersection_async<bool_t>(const Tensor<bool_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bool_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_t>(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_bf16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp64_t>(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<int64_t>(const Tensor<int64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<int64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp16_t>(const Tensor<fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<bf16_t>(const Tensor<bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bf16_t> &dst,
        const std::vector<Index> &dst_offset);

// Explicit instantiation
template
void copy_intersection<bool_t>(const Tensor<bool_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bool_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_t>(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_bf16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp64_t>(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<int64_t>(const Tensor<int64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<int64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp16_t>(const Tensor<fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<bf16_t>(const Tensor<bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bf16_t> &dst,
        const std::vector<Index> &dst_offset);

} // namespace nntile::tensor
