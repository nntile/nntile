/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tensor<T> (out-of-place version)
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm_slice.hh"
#include "nntile/tile/norm_slice.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void norm_slice_async(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst, Index axis, int redux)
{
    // Check dimensions
    if(src1.ndim-1 != src2.ndim)
    {
        throw std::runtime_error("src1.ndim-1 != src2.ndim");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
    }
    // Treat special case of src1.ndim=0
    if(src1.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    Index ndim = src1.ndim;
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src1.ndim)
    {
        throw std::runtime_error("axis >= src1.ndim");
    }
    // Check shapes of src1, src2 and dst
    for(Index i = 0; i < axis; i++)
    {
        if(src1.shape[i] != src2.shape[i])
        {
            throw std::runtime_error("src1.shape[i] != src2.shape[i]");
        }
        if(src1.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i]");
        }
        if(src1.basetile_shape[i] != src2.basetile_shape[i])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "src2.basetile_shape[i]");
        }
        if(src1.basetile_shape[i] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "dst.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src1.shape[i] != src2.shape[i-1])
        {
            throw std::runtime_error("src1.shape[i] != src2.shape[i-1]");
        }
        if(src1.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i-1]");
        }
        if(src1.basetile_shape[i] != src2.basetile_shape[i-1])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "src2.basetile_shape[i-1]");
        }
        if(src1.basetile_shape[i] != dst.basetile_shape[i-1])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "dst.basetile_shape[i-1]");
        }
    }
    // Do actual calculations
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto dst_tile = dst.get_tile(i);
        auto src2_tile = src2.get_tile(i);
        // Obtain indices of applicable source tiles
        auto dst_tile_index = dst.grid.linear_to_index(i);
        std::vector<Index> src1_tile_index(src1.ndim);
        for(Index j = 0, k = 0; j < src1.ndim; ++j)
        {
            if(j == axis)
            {
                src1_tile_index[axis] = 0;
                continue;
            }
            src1_tile_index[j] = dst_tile_index[k];
            ++k;
        }
        // Launch kernel for each appropriate source tile
        for(Index j = 0; j < src1.grid.shape[axis]; ++j)
        {
            src1_tile_index[axis] = j;
            Index src1_tile_offset = src1.grid.index_to_linear(src1_tile_index);
            auto src1_tile = src1.get_tile(src1_tile_offset);
            tile::norm_slice_async<T>(alpha, src1_tile, beta, src2_tile,
                    dst_tile, axis, redux);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void norm_slice(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis, int redux)
{
    norm_slice_async<T>(alpha, src1, beta, src2, dst, axis, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_slice_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, Scalar beta,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void norm_slice<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis, int redux);

template
void norm_slice<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, Scalar beta,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst, Index axis, int redux);

} // namespace nntile::tensor
