/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm_slice_inplace.cc
 * Euclidean norms of fibers into a slice of a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm_slice_inplace.hh"
#include "nntile/tile/norm_slice_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void norm_slice_inplace_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const Tensor<T> &dst, Index axis, int redux)
{
    // Check dimensions
    if(src.ndim-1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim-1 != dst.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    Index ndim = src.ndim;
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and dst
    for(Index i = 0; i < axis; i++)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
        if(src.basetile_shape[i] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "dst.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i-1]");
        }
        if(src.basetile_shape[i] != dst.basetile_shape[i-1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "dst.basetile_shape[i-1]");
        }
    }
    // Do actual calculations
    constexpr Scalar one = 1.0;
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto dst_tile = dst.get_tile(i);
        // Obtain indices of applicable source tiles
        auto dst_tile_index = dst.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src.ndim);
        for(Index j = 0, k = 0; j < src.ndim; ++j)
        {
            if(j == axis)
            {
                src_tile_index[axis] = 0;
                continue;
            }
            src_tile_index[j] = dst_tile_index[k];
            ++k;
        }
        // Launch kernel for each appropriate source tile
        for(Index j = 0; j < src.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            Index src_tile_offset = src.grid.index_to_linear(src_tile_index);
            auto src_tile = src.get_tile(src_tile_offset);
            if(j == 0)
            {
                tile::norm_slice_inplace_async<T>(alpha, src_tile, beta,
                        dst_tile, axis, redux);
            }
            else
            {
                tile::norm_slice_inplace_async<T>(alpha, src_tile, one,
                        dst_tile, axis, redux);
            }
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void norm_slice_inplace(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        Index axis, int redux)
{
    norm_slice_inplace_async<T>(alpha, src, beta, dst, axis, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_slice_inplace_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        Scalar beta, const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        Scalar beta, const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void norm_slice_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst, Index axis, int redux);

} // namespace nntile::tensor
